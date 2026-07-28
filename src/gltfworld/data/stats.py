"""``gltfworld stats``: a sanity/statistics report for a packed dataset (see
``gltfworld.data.pack``), the thing the PRE-TRAINING GATE actually checks
before any model training starts.

Everything here reads *only* the packed ``safetensors`` file + its
``pack_meta.json`` sidecar -- no re-parsing of the source GLBs -- since the
tensor contract (``gltfworld.scene.contract``) plus ``pack_dataset``'s
``ground_top_y`` (see ``pack.py``'s ``_static_top_y``) already carry
everything needed: per-object pose/velocity/shape/size/mass/friction, and
where the ground's top surface is.

Documented approximations
--------------------------

- **Ground penetration**: uses the same orientation-aware "support
  function" as ``gltfworld.datagen.sample.object_support_offset``
  (exact for sphere/box/cylinder), evaluated along world ``-Y`` to find
  each dynamic object's lowest point at each frame, compared against the
  packed ``ground_top_y`` (see ``pack.py``). Steady state = the last 20% of
  an episode's frames (matches ``tests/test_episode_pipeline.py``'s
  convention, see DESIGN.md "Ground-contact tolerances").
- **Energy trend**: "total energy" here is translational kinetic
  (``0.5 * m * |v|^2``) plus gravitational potential (``m * |g| * y``)
  only -- **rotational kinetic energy is deliberately excluded** (the
  tensor contract carries angular velocity but not a per-object inertia
  tensor, and approximating one from shape+size+mass would be a much
  bigger source of error than just omitting a term that's usually small
  next to translational KE for gltfworld's slow-spinning small objects).
  This is a coarse "is the system dissipating energy through
  contacts, not blowing up" sanity signal, not an energy-conservation
  check. The raw per-frame energy is smoothed with a simple moving
  average (window 5 frames, reducing contact-transient noise) and an
  episode is counted as "non-increasing" if a least-squares linear fit to
  the smoothed series has a non-positive slope, allowing a small positive
  tolerance (default 2% of the episode's own smoothed energy range) so
  floating-point/contact noise on an already-flat trend doesn't get
  misclassified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from gltfworld.data.pack import SPLIT_NAMES, _pack_meta_path
from gltfworld.scene.contract import CATEGORY_TO_CLASS_ID, CLASS_ID_TO_CATEGORY, SHAPE_ORDER

_STEADY_STATE_FRACTION = 0.2
_PENETRATION_STEADY_MM = 5.0
_PENETRATION_TRANSIENT_MM = 100.0
_ENERGY_SMOOTH_WINDOW = 5
_ENERGY_SLOPE_TOL_FRACTION = 0.02


def _range_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"min": None, "mean": None, "max": None, "count": 0}
    return {"min": float(np.min(x)), "mean": float(np.mean(x)), "max": float(np.max(x)), "count": int(x.size)}


def _support_offset_batch(shape_idx: np.ndarray, size: np.ndarray, quat_xyzw: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Vectorized version of
    ``gltfworld.datagen.sample.object_support_offset``'s support function,
    over arbitrary leading batch dims. ``shape_idx`` in ``{0:sphere,
    1:box, 2:cylinder}`` (matches ``gltfworld.scene.contract.SHAPE_ORDER``),
    ``size`` ``(..., 3)``, ``quat_xyzw`` ``(..., 4)``, ``direction`` ``(3,)``
    a world-frame unit vector.
    """
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    # Rotation matrix rows (world <- local); we need R.T @ direction, i.e.
    # direction expressed in the object's local frame.
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    dx, dy, dz = direction
    # R.T @ direction: column c of R.T is row c of R -> (R.T @ d)_i = sum_j R[j,i] * d[j]
    d_local_x = r00 * dx + r10 * dy + r20 * dz
    d_local_y = r01 * dx + r11 * dy + r21 * dz
    d_local_z = r02 * dx + r12 * dy + r22 * dz

    sphere_offset = size[..., 0]
    box_offset = np.abs(d_local_x) * size[..., 0] + np.abs(d_local_y) * size[..., 1] + np.abs(d_local_z) * size[..., 2]
    radius = size[..., 0]
    half_height = size[..., 1]
    cyl_offset = radius * np.hypot(d_local_x, d_local_z) + half_height * np.abs(d_local_y)

    offset = np.where(shape_idx == 0, sphere_offset, np.where(shape_idx == 1, box_offset, cyl_offset))
    return offset


def _ground_penetration_and_energy(
    states: np.ndarray,
    mask: np.ndarray,
    globals_: np.ndarray,
    ground_top_y: float,
    ground_footprint: dict | None,
) -> dict:
    """Per-episode ground-penetration (steady-state + transient) and energy
    trend, vectorized over the whole packed dataset.

    ``states``: ``(E, T, N, D)`` float32. ``mask``: ``(E, N)`` bool.
    ``globals_``: ``(E, G)`` float32 (``gravity`` in ``[:, 0:3]``).

    ``ground_footprint`` (from ``pack_meta.json``, see ``pack.py``'s
    ``_static_footprint_half_extents``): when given, (x, z) frames outside
    the ground's finite footprint are EXCLUDED from the penetration checks
    below -- ``wm-scenes-v1``'s ground is a *finite* plate (DESIGN.md
    "Finite ground plate"), and an object that has rolled/bounced off its
    edge is genuinely free-falling with nothing underneath it, not
    "penetrating" anything. Those excluded frames are still reported
    separately (``off_plate_object_frame_fraction`` /
    ``episodes_with_departure_fraction``) so this scope boundary stays
    visible instead of silently vanishing into a filtered-out denominator.
    """
    e, t, n, d = states.shape
    pos = states[..., 0:3]
    quat = states[..., 3:7]
    lin_vel = states[..., 7:10]
    shape_onehot = states[..., 13:16]
    size = states[..., 16:19]
    log_mass = states[..., 19]
    mass = np.exp(log_mass)

    shape_idx = np.argmax(shape_onehot, axis=-1)  # (E, T, N)

    down = np.array([0.0, -1.0, 0.0])
    support_down = _support_offset_batch(shape_idx, size, quat, down)  # (E, T, N)
    lowest_y = pos[..., 1] - support_down  # (E, T, N)
    penetration_mm = (ground_top_y - lowest_y) * 1000.0  # positive = below ground

    mask_b = mask[:, None, :]  # (E, 1, N) broadcast over T

    on_plate = np.ones((e, t, n), dtype=bool)
    off_plate_fraction = 0.0
    episodes_with_departure_fraction = 0.0
    if ground_footprint is not None:
        dx = np.abs(pos[..., 0] - ground_footprint["center_x"])
        dz = np.abs(pos[..., 2] - ground_footprint["center_z"])
        on_plate = (dx <= ground_footprint["half_extent_x"]) & (dz <= ground_footprint["half_extent_z"])
        valid_object_frame_count = int(np.sum(np.broadcast_to(mask_b, (e, t, n))))
        off_plate_fraction = float(np.sum((~on_plate) & mask_b) / max(valid_object_frame_count, 1))
        departed = np.any((~on_plate) & mask_b, axis=(1, 2))  # (E,)
        episodes_with_departure_fraction = float(np.mean(departed))

    # Mask out padding rows AND off-plate frames with -inf so they never win a max.
    include = mask_b & on_plate
    penetration_masked = np.where(include, penetration_mm, -np.inf)

    # If an object left the plate for its entire episode (rare, but
    # possible for e.g. a fast roller), fall back to -inf -> handled by
    # nan-safe reduction below (max of all -inf is -inf; excluded from
    # the range/threshold stats via isfinite masking).
    transient_max_mm = np.max(penetration_masked, axis=(1, 2))  # (E,) worst penetration, any frame

    steady_start = int(round(t * (1.0 - _STEADY_STATE_FRACTION)))
    steady_penetration = penetration_masked[:, steady_start:, :]
    steady_max_mm = np.max(steady_penetration, axis=(1, 2)) if steady_penetration.shape[1] > 0 else transient_max_mm

    finite_transient = np.isfinite(transient_max_mm)
    finite_steady = np.isfinite(steady_max_mm)

    gravity_y = globals_[:, 1]  # (E,)
    ke = 0.5 * mass * np.sum(lin_vel**2, axis=-1)  # (E, T, N)
    pe = mass * np.abs(gravity_y)[:, None, None] * pos[..., 1]  # (E, T, N)
    energy_per_object = np.where(mask_b, ke + pe, 0.0)
    total_energy = np.sum(energy_per_object, axis=-1)  # (E, T)

    window = min(_ENERGY_SMOOTH_WINDOW, t)
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=total_energy)

    frames = np.arange(smoothed.shape[1], dtype=np.float64)
    non_increasing = np.zeros(e, dtype=bool)
    for i in range(e):
        series = smoothed[i]
        if series.shape[0] < 2:
            non_increasing[i] = True
            continue
        slope, _ = np.polyfit(frames, series, 1)
        energy_range = float(np.max(series) - np.min(series))
        tol = max(_ENERGY_SLOPE_TOL_FRACTION * energy_range / max(frames[-1], 1.0), 1e-9)
        non_increasing[i] = slope <= tol

    return {
        "transient_max_penetration_mm": _range_stats(transient_max_mm[finite_transient]),
        "steady_state_max_penetration_mm": _range_stats(steady_max_mm[finite_steady]),
        "steady_state_within_5mm_fraction": float(
            np.mean(steady_max_mm[finite_steady] <= _PENETRATION_STEADY_MM) if finite_steady.any() else float("nan")
        ),
        "transient_within_100mm_fraction": float(
            np.mean(transient_max_mm[finite_transient] <= _PENETRATION_TRANSIENT_MM)
            if finite_transient.any()
            else float("nan")
        ),
        "energy_non_increasing_fraction": float(np.mean(non_increasing)),
        "off_plate_object_frame_fraction": off_plate_fraction,
        "episodes_with_departure_fraction": episodes_with_departure_fraction,
    }


def compute_stats(pack_file: str | Path) -> dict:
    """Compute the full stats report for the packed dataset at ``pack_file``."""
    pack_file = Path(pack_file)
    tensors = load_file(str(pack_file))
    meta_path = _pack_meta_path(pack_file)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    states = tensors["states"]
    mask = tensors["mask"]
    class_ids = tensors["class_ids"]
    globals_ = tensors["globals"]
    split_id = tensors["split_id"]

    e, t, n_max, d = states.shape
    num_transitions = int(e * (t - 1))

    # `mask` is (E, N_max); broadcast to select valid object *rows* across all T frames.
    mask_bt = np.broadcast_to(mask[:, None, :], (e, t, n_max))
    valid_states = states[mask_bt]  # (num_valid_frames_x_objects, D)

    shape_idx = np.argmax(valid_states[:, 13:16], axis=-1)
    shape_counts = {SHAPE_ORDER[i]: int(np.sum(shape_idx == i)) for i in range(len(SHAPE_ORDER))}

    valid_class_ids = class_ids[mask]  # (num_valid_object_instances,) -- one row per (episode, object)
    class_hist = {
        CLASS_ID_TO_CATEGORY.get(int(c), f"unknown_{c}"): int(np.sum(valid_class_ids == c))
        for c in sorted(set(int(c) for c in valid_class_ids.tolist()))
    }

    pos = valid_states[:, 0:3]
    lin_vel = valid_states[:, 7:10]
    ang_vel = valid_states[:, 10:13]
    mass = np.exp(valid_states[:, 19])
    friction = valid_states[:, 20]

    lin_speed = np.linalg.norm(lin_vel, axis=-1)
    ang_speed = np.linalg.norm(ang_vel, axis=-1)

    nan_inf_count = int(np.sum(~np.isfinite(states)) + np.sum(~np.isfinite(globals_)))

    split_counts = {name: int(np.sum(split_id == i)) for i, name in enumerate(SPLIT_NAMES)}

    ground_top_y = None
    if meta.get("ground_top_y", {}).get("mean") is not None:
        ground_top_y = meta["ground_top_y"]["mean"]
    ground_footprint = meta.get("ground_footprint")

    physics_ok = ground_top_y is not None
    ground_stats = (
        _ground_penetration_and_energy(states, mask, globals_, ground_top_y, ground_footprint)
        if physics_ok
        else {"error": "no ground_top_y recorded in pack_meta.json"}
    )

    return {
        "pack_file": str(pack_file),
        "episode_count": int(e),
        "transition_count": num_transitions,
        "frames_per_episode": int(t),
        "n_max": int(n_max),
        "d": int(d),
        "split_counts": split_counts,
        "per_shape_counts": shape_counts,
        "class_id_histogram": class_hist,
        "position_range": {axis: _range_stats(pos[:, i]) for i, axis in enumerate(("x", "y", "z"))},
        "linear_velocity_range": {axis: _range_stats(lin_vel[:, i]) for i, axis in enumerate(("x", "y", "z"))},
        "linear_speed_range": _range_stats(lin_speed),
        "angular_velocity_range": {axis: _range_stats(ang_vel[:, i]) for i, axis in enumerate(("x", "y", "z"))},
        "angular_speed_range": _range_stats(ang_speed),
        "mass_range": _range_stats(mass),
        "friction_range": _range_stats(friction),
        "nan_inf_count": nan_inf_count,
        "ground_penetration_and_energy": ground_stats,
    }


def format_human(report: dict) -> str:
    lines = []
    lines.append(f"dataset: {report['pack_file']}")
    lines.append(f"episodes: {report['episode_count']}  transitions: {report['transition_count']}  T: {report['frames_per_episode']}  N_max: {report['n_max']}  D: {report['d']}")
    lines.append(f"split sizes: {report['split_counts']}")
    lines.append(f"per-shape counts: {report['per_shape_counts']}")
    lines.append(f"class-id histogram: {report['class_id_histogram']}")

    def _range_line(name, r):
        if r["count"] == 0:
            return f"  {name}: (no data)"
        return f"  {name}: min={r['min']:.4g} mean={r['mean']:.4g} max={r['max']:.4g} (n={r['count']})"

    lines.append("position range (m):")
    for axis, r in report["position_range"].items():
        lines.append(_range_line(axis, r))
    lines.append("linear velocity range (m/s):")
    for axis, r in report["linear_velocity_range"].items():
        lines.append(_range_line(axis, r))
    lines.append(_range_line("linear speed", report["linear_speed_range"]))
    lines.append("angular velocity range (rad/s):")
    for axis, r in report["angular_velocity_range"].items():
        lines.append(_range_line(axis, r))
    lines.append(_range_line("angular speed", report["angular_speed_range"]))
    lines.append(_range_line("mass (kg)", report["mass_range"]))
    lines.append(_range_line("friction", report["friction_range"]))
    lines.append(f"NaN/Inf count: {report['nan_inf_count']} (must be 0)")

    ge = report["ground_penetration_and_energy"]
    if "error" in ge:
        lines.append(f"ground penetration / energy trend: unavailable ({ge['error']})")
    else:
        lines.append("ground penetration / energy trend:")
        lines.append(f"  {_range_line('steady-state max penetration (mm)', ge['steady_state_max_penetration_mm'])}")
        lines.append(f"  {_range_line('transient max penetration (mm)', ge['transient_max_penetration_mm'])}")
        lines.append(
            f"  fraction of episodes with steady-state penetration <= {_PENETRATION_STEADY_MM}mm: "
            f"{ge['steady_state_within_5mm_fraction']:.4f}"
        )
        lines.append(
            f"  fraction of episodes with transient penetration <= {_PENETRATION_TRANSIENT_MM}mm: "
            f"{ge['transient_within_100mm_fraction']:.4f}"
        )
        lines.append(
            f"  fraction of episodes with non-increasing smoothed total energy: "
            f"{ge['energy_non_increasing_fraction']:.4f}"
        )
        lines.append(
            f"  off-plate object-frames (excluded from penetration stats above, finite plate -- see DESIGN.md): "
            f"{ge['off_plate_object_frame_fraction']:.4%}"
        )
        lines.append(
            f"  episodes where >=1 object left the plate at some point: "
            f"{ge['episodes_with_departure_fraction']:.4%}"
        )
    return "\n".join(lines)
