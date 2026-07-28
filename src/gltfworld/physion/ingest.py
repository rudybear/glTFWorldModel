"""Ingest for the Physion benchmark's "Core" test archive (PhysionTest-Core,
``Physion.zip``, S3 bucket ``physics-benchmarking-neurips2021-dataset``).

This is reconnaissance/ingest only -- no OCP evaluation logic lives here (see
V8's own milestone scope). See ``docs/PHYSION.md`` for the full archive
format writeup (directory layout, what ground truth this "Core" tier does and
doesn't carry vs. the full per-scenario HDF5 releases, verified download URLs
+ sizes) this module is built from.

Expected extracted layout (``root``, default
``data/external/physion/extracted/Physion``, produced by unzipping
``data/external/physion/Physion.zip`` -- both paths are gitignored, this
module never downloads or extracts anything itself)::

    Physion/
      labels.csv                                  # trial_id -> "True"/"False" OCP outcome
      {Scenario}/mp4s/{trial_id}.mp4               # human-viewed video (agent/patient random colors)
      {Scenario}/mp4s-redyellow/{trial_id_ry}.mp4  # model-input video (agent=red, patient=yellow)
      {Scenario}/maps/{trial_stem}_map.png         # single-frame agent(red)/patient(yellow) map

``Scenario`` in ``SCENARIOS`` below. ``trial_id`` is ``labels.csv``'s row key
and also the ``mp4s`` filename stem (e.g.
``pilot_dominoes_4mid_boxroom_2_0002_img``); it is globally unique across all
1200 trials in the real archive (verified directly against the extracted
data, not assumed), so no ``(scenario, trial_id)`` composite key is needed --
``scenario`` is carried anyway since it isn't recoverable from ``trial_id``
alone.

**``mp4s-redyellow`` filename derivation.** The redyellow video for a given
``trial_id`` is not simply the same filename in a different directory: a
literal ``"-redyellow"`` is spliced in immediately before the trailing
4-digit trial index, e.g. ``..._1_dis_1_occ_0000_img.mp4`` (``mp4s``) <->
``..._1_dis_1_occ-redyellow_0000_img.mp4`` (``mp4s-redyellow``). Verified
against all 1200 trials in the real archive (100% match, zero exceptions to
the "insert before the trailing ``_NNNN``" rule).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/external/physion/extracted/Physion")

# Fixed, deterministic scenario order (alphabetical; matches the archive's
# own top-level directory names). Drape is the one non-rigid scenario (cloth
# draping); kept in the same index as the other seven -- see docs/PHYSION.md
# for the rigid/non-rigid distinction, which matters for the HDF5
# state-based track (V8 option b), not for this pixel-level index.
SCENARIOS = ("Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support")

_TRAILING_INDEX_RE = re.compile(r"^(.*)_(\d{4})$")


def _insert_redyellow(stem: str) -> str:
    """``..._1_dis_1_occ_0000`` -> ``..._1_dis_1_occ-redyellow_0000``."""
    m = _TRAILING_INDEX_RE.match(stem)
    if not m:
        raise ValueError(f"expected a trailing 4-digit trial index in {stem!r}")
    return f"{m.group(1)}-redyellow_{m.group(2)}"


@dataclass(frozen=True)
class PhysionTrial:
    """One Physion Core trial. ``label`` is the OCP ground truth: whether the
    patient (yellow) object is ever contacted by the agent (red) object,
    per ``labels.csv``'s ``"ground truth outcome"`` column.
    """

    scenario: str
    trial_id: str
    video_path: Path
    redyellow_path: Path
    map_path: Path
    label: bool


def _parse_label(value: str) -> bool:
    value = value.strip()
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unrecognized labels.csv value {value!r} (expected 'True'/'False')")


class PhysionIndex:
    """Enumerates every trial in an extracted PhysionTest-Core archive.

    Only scenarios whose directory actually exists under ``root`` are
    indexed (a partially-extracted archive doesn't error, it just yields
    fewer trials) -- but ``labels.csv`` itself is required, since every
    trial needs its OCP ground truth.

    Iteration/indexing order is deterministic: scenarios in ``SCENARIOS``
    order, trials within a scenario sorted by ``trial_id``.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        labels_path = self.root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"{labels_path} not found -- expected an extracted PhysionTest-Core "
                f"archive at {self.root} (see docs/PHYSION.md)"
            )
        labels = self._load_labels(labels_path)

        trials: list[PhysionTrial] = []
        for scenario in SCENARIOS:
            mp4_dir = self.root / scenario / "mp4s"
            if not mp4_dir.is_dir():
                continue
            ry_dir = self.root / scenario / "mp4s-redyellow"
            map_dir = self.root / scenario / "maps"
            for video_path in sorted(mp4_dir.glob("*.mp4")):
                trial_id = video_path.stem  # e.g. "..._0000_img"
                if trial_id not in labels:
                    raise KeyError(f"{video_path} has no labels.csv entry for trial_id {trial_id!r}")
                if not trial_id.endswith("_img"):
                    raise ValueError(f"expected trial_id {trial_id!r} to end with '_img'")
                base_stem = trial_id[: -len("_img")]
                redyellow_path = ry_dir / f"{_insert_redyellow(base_stem)}_img.mp4"
                map_path = map_dir / f"{base_stem}_map.png"
                trials.append(
                    PhysionTrial(
                        scenario=scenario,
                        trial_id=trial_id,
                        video_path=video_path,
                        redyellow_path=redyellow_path,
                        map_path=map_path,
                        label=labels[trial_id],
                    )
                )
        trials.sort(key=lambda t: (SCENARIOS.index(t.scenario), t.trial_id))
        self.trials: list[PhysionTrial] = trials
        self._by_id = {t.trial_id: t for t in trials}

    @staticmethod
    def _load_labels(labels_path: Path) -> dict[str, bool]:
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "ground truth outcome" not in reader.fieldnames:
                raise ValueError(f"{labels_path} missing 'ground truth outcome' column")
            id_field = reader.fieldnames[0]  # unnamed first column holding the trial id
            return {row[id_field]: _parse_label(row["ground truth outcome"]) for row in reader}

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)

    def __getitem__(self, index: int) -> PhysionTrial:
        return self.trials[index]

    def by_id(self, trial_id: str) -> PhysionTrial:
        return self._by_id[trial_id]

    def scenario_counts(self) -> dict[str, int]:
        """``{scenario: trial_count}`` for scenarios actually present, in
        ``SCENARIOS`` order."""
        counts: dict[str, int] = {}
        for t in self.trials:
            counts[t.scenario] = counts.get(t.scenario, 0) + 1
        return counts


def load_frames(
    trial: PhysionTrial | str | Path,
    max_frames: int | None = None,
    stride: int = 1,
    use_redyellow: bool = False,
) -> np.ndarray:
    """Decode an mp4 into an in-memory ``uint8`` frame stack, ``(T, H, W, 3)``.

    ``trial`` may be a ``PhysionTrial`` (uses ``video_path`` or
    ``redyellow_path`` per ``use_redyellow``), or a path directly. Frames are
    read at ``stride`` (every ``stride``-th frame, 1-based-count semantics:
    ``stride=1`` keeps every frame), stopping once ``max_frames`` decoded
    frames have been kept (``None`` means the whole video). Dependency-light:
    only ``imageio`` (already a project dependency via the ``ml`` extra,
    ``imageio[ffmpeg]``), no ``h5py``/``opencv``/video-specific extras.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if isinstance(trial, PhysionTrial):
        path = trial.redyellow_path if use_redyellow else trial.video_path
    else:
        path = Path(trial)

    import imageio  # local import: only needed by this function, keeps module import light

    frames: list[np.ndarray] = []
    reader = imageio.get_reader(str(path))
    try:
        for i, frame in enumerate(reader):
            if i % stride != 0:
                continue
            frames.append(np.asarray(frame, dtype=np.uint8))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"decoded zero frames from {path}")
    return np.stack(frames, axis=0)
