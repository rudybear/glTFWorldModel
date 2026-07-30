"""V9-prep physics sanity (MuJoCo-backed, needs the ``sim`` extra):

- door pushed with +torque swings open (monotonically, up to its peak),
  settles within its joint limits;
- drawer slides within its travel limits under the same scripted push;
- **the articulation consistency check**: the moving part's recorded pose
  must equal ``anchor`` composed with the joint transform implied by the
  recorded ``joint_pos`` at every step, both for a freshly simulated
  in-memory episode and after a real GLB save/load round trip (mirroring
  this project's existing "provenance" testing pattern -- see
  ``tests/test_provenance.py``).

``axis`` is pinned (not left to the general sampler's random choice) for the
monotonic-opening checks specifically, per
``gltfworld.datagen.articulated``'s own documented reasoning: a vertical
hinge axis (door) and a horizontal slide axis (drawer) are gravity-torque-
/gravity-force-decoupled, giving a reliably reproducible trajectory shape --
other axis choices are physically valid but may have gravity assisting or
opposing the push (a real effect, not a bug; see DESIGN.md's V9-prep report).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.datagen.articulated import sample_articulated_scene, simulate_articulated
from gltfworld.scene.convert import episode_from_gltf, episode_to_gltf, load_episode, save_episode
from gltfworld.scene.episode import Episode

_T = 150
_HZ = 30.0
_SEEDS = [1, 17, 42, 101, 256]


def _rotate_about_cardinal_axis(v: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula, specialized to a cardinal (X/Y/Z) axis."""
    k = np.zeros(3)
    k[axis] = 1.0
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1.0 - np.cos(angle))


def _axis_angle_quat(axis: int, angle: float) -> np.ndarray:
    k = np.zeros(3)
    k[axis] = 1.0
    half = angle / 2.0
    return np.array([k[0] * np.sin(half), k[1] * np.sin(half), k[2] * np.sin(half), np.cos(half)])


def _build_episode(seed: int, kind: str, axis: int) -> Episode:
    sampled = sample_articulated_scene(seed, kind=kind, axis=axis)
    series = simulate_articulated(sampled, T=_T, record_hz=_HZ)
    return Episode(scene=sampled.scene, series=series)


# --- door: monotonic open, settle within limits ----------------------------


@pytest.mark.parametrize("seed", _SEEDS)
def test_door_pushed_opens_and_settles_within_limits(seed):
    ep = _build_episode(seed, kind="door", axis=1)  # vertical hinge: gravity-torque-free
    art = ep.scene.articulations[0]
    jp = ep.series.joint_pos[:, 0]

    peak_index = int(np.argmax(jp))
    assert peak_index > 0, "push should have moved the door at all"

    # Monotonically non-decreasing (small tolerance for integrator noise)
    # from the start up to its peak -- "swings open monotonically".
    diffs = np.diff(jp[: peak_index + 1])
    assert np.all(diffs >= -2e-3), f"non-monotonic opening: min step {diffs.min():.4f}"

    # Settled (low variance) by the end of the episode -- "settles".
    tail = jp[-15:]
    assert tail.std() < 5e-3, f"still oscillating at episode end: tail std {tail.std():.4f}"

    # "within limits": final resting position inside [min, max], generous
    # tolerance for the soft-constraint solver's own small, documented
    # steady-state penetration (see DESIGN.md's V9-prep report -- measured
    # up to ~0.01 rad past a hard stop, analogous to V3's ground-contact
    # tolerance note).
    tol = 0.05 * (art.max - art.min) + 0.02
    assert tail.mean() >= art.min - tol
    assert tail.mean() <= art.max + tol

    # Sanity bound on the transient overshoot itself -- catches a real bug
    # (e.g. the degree/radian MJCF mixup found during development, see
    # DESIGN.md) that would blow the peak far past the limit.
    assert jp.max() <= art.max * 1.5 + 0.1


# --- drawer: slides within travel ------------------------------------------


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("axis", [0, 2])  # horizontal slide axes: gravity-force-perpendicular
def test_drawer_pushed_slides_within_travel(seed, axis):
    ep = _build_episode(seed, kind="drawer", axis=axis)
    art = ep.scene.articulations[0]
    jp = ep.series.joint_pos[:, 0]

    tol = 0.05 * (art.max - art.min) + 0.01
    assert jp.min() >= art.min - tol
    assert jp.max() <= art.max + tol

    # Real net displacement -- the push actually moved it, not a no-op.
    assert jp.max() - jp[0] > 0.3 * (art.max - art.min)

    tail = jp[-15:]
    assert tail.std() < 5e-3


# --- THE articulation consistency check ------------------------------------


def _check_articulation_consistency(ep: Episode, atol_pos: float = 0.03, atol_rot: float = 0.03) -> None:
    """part_pose(t) must equal anchor (a fixed point, since base is static)
    composed with the joint transform implied by the recorded joint_pos(t),
    at every recorded step -- reconstructed purely from the anchor/axis
    metadata (``ArticulatedSpec``) and the recorded ``poses``/``joint_pos``,
    with no privileged access to the simulator's own internal state."""
    art = ep.scene.articulations[0]
    obj_ids = [o.object_id for o in ep.scene.objects]
    base_index = obj_ids.index(art.base_object_id)
    part_index = obj_ids.index(art.part_object_id)

    poses = ep.series.poses
    jp = ep.series.joint_pos[:, 0]
    axis_vec = np.zeros(3)
    axis_vec[art.axis] = 1.0
    anchor = art.anchor.astype(np.float64)

    # Sanity: base is static -- its recorded pose must never move.
    base_pos0 = poses[0, base_index, 0:3]
    # np.broadcast_to (not bare assert_allclose(a, b)): this numpy version's
    # assert_allclose enforces exact shape equality even with strict=False
    # (no longer auto-broadcasts (150, 3) against (3,)), so broadcast by hand.
    np.testing.assert_allclose(poses[:, base_index, 0:3], np.broadcast_to(base_pos0, poses[:, base_index, 0:3].shape), atol=1e-4)

    part_pos0 = poses[0, part_index, 0:3].astype(np.float64)
    if art.joint_type == "revolute":
        rest_offset = _rotate_about_cardinal_axis(part_pos0 - anchor, art.axis, -float(jp[0]))
    else:
        rest_offset = part_pos0 - anchor - axis_vec * float(jp[0])

    max_pos_err = 0.0
    max_rot_err = 0.0
    for t in range(len(jp)):
        angle = float(jp[t])
        if art.joint_type == "revolute":
            pred_pos = anchor + _rotate_about_cardinal_axis(rest_offset, art.axis, angle)
            pred_rot = _axis_angle_quat(art.axis, angle)
        else:
            pred_pos = anchor + rest_offset + axis_vec * angle
            pred_rot = np.array([0.0, 0.0, 0.0, 1.0])

        actual_pos = poses[t, part_index, 0:3].astype(np.float64)
        actual_rot = poses[t, part_index, 3:7].astype(np.float64)

        pos_err = float(np.linalg.norm(pred_pos - actual_pos))
        # quaternion double-cover: q and -q are the same rotation
        rot_err = float(min(np.linalg.norm(pred_rot - actual_rot), np.linalg.norm(pred_rot + actual_rot)))

        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)

    assert max_pos_err < atol_pos, f"articulation consistency: max position error {max_pos_err:.4f}m"
    assert max_rot_err < atol_rot, f"articulation consistency: max rotation error {max_rot_err:.4f}"


@pytest.mark.parametrize("kind,axis", [("door", 1), ("drawer", 0)])
@pytest.mark.parametrize("seed", _SEEDS)
def test_articulation_consistency_in_memory(seed, kind, axis):
    ep = _build_episode(seed, kind=kind, axis=axis)
    _check_articulation_consistency(ep)


@pytest.mark.parametrize("kind,axis", [("door", 1), ("drawer", 0)])
def test_articulation_consistency_after_glb_roundtrip(tmp_path, kind, axis):
    """Provenance-style check (see tests/test_provenance.py): the same
    invariant must hold using the tensors/arrays actually read back off a
    real .glb file on disk, not just the in-memory Episode."""
    ep = _build_episode(seed=7, kind=kind, axis=axis)

    path = tmp_path / f"articulated_{kind}.glb"
    save_episode(ep, path)
    decoded = load_episode(path)

    _check_articulation_consistency(decoded)

    # Also confirm the in-memory encode/decode path (no file I/O) agrees.
    decoded_in_memory = episode_from_gltf(episode_to_gltf(ep))
    _check_articulation_consistency(decoded_in_memory)
