"""Episode = SceneState + StateSeries: the unit gltfworld ships as one GLB.

``StateSeries`` is the "how does it move over time" half of an episode: a
shared time axis plus per-object poses (required) and optional per-object
velocities/pose-variance and per-episode actions. All arrays are float32 and
shape-validated in ``__post_init__`` against the (T, N) implied by
``times``/``poses`` so a malformed series fails fast, before it ever reaches
the glTF codec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import SceneState


def _f32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


@dataclass
class StateSeries:
    times: np.ndarray  # (T,) float32 seconds
    poses: np.ndarray  # (T, N, 7) float32: xyz + quat xyzw
    lin_vel: np.ndarray | None = None  # (T, N, 3) float32
    ang_vel: np.ndarray | None = None  # (T, N, 3) float32
    actions: np.ndarray | None = None  # (T, A) float32
    pose_var: np.ndarray | None = None  # (T, N, 7) float32

    def __post_init__(self) -> None:
        self.times = _f32(self.times)
        if self.times.ndim != 1:
            raise ValueError(f"times must be 1D (T,), got shape {self.times.shape}")
        t = self.times.shape[0]

        self.poses = _f32(self.poses)
        if self.poses.ndim != 3 or self.poses.shape[0] != t or self.poses.shape[2] != 7:
            raise ValueError(f"poses must have shape (T={t}, N, 7), got {self.poses.shape}")
        n = self.poses.shape[1]

        for attr in ("lin_vel", "ang_vel"):
            value = getattr(self, attr)
            if value is not None:
                value = _f32(value)
                if value.shape != (t, n, 3):
                    raise ValueError(
                        f"{attr} must have shape (T={t}, N={n}, 3), got {value.shape}"
                    )
                setattr(self, attr, value)

        if self.actions is not None:
            self.actions = _f32(self.actions)
            if self.actions.ndim != 2 or self.actions.shape[0] != t:
                raise ValueError(f"actions must have shape (T={t}, A), got {self.actions.shape}")

        if self.pose_var is not None:
            self.pose_var = _f32(self.pose_var)
            if self.pose_var.shape != (t, n, 7):
                raise ValueError(
                    f"pose_var must have shape (T={t}, N={n}, 7), got {self.pose_var.shape}"
                )

    @property
    def num_frames(self) -> int:
        return self.times.shape[0]

    @property
    def num_objects(self) -> int:
        return self.poses.shape[1]


@dataclass
class Episode:
    scene: SceneState
    series: StateSeries

    def __post_init__(self) -> None:
        n_scene = len(self.scene.objects)
        n_series = self.series.num_objects
        if n_scene != n_series:
            raise ValueError(
                f"scene declares {n_scene} objects but series carries {n_series}"
            )

    @classmethod
    def single_frame(
        cls,
        scene: SceneState,
        poses,
        *,
        time: float = 0.0,
        lin_vel=None,
        ang_vel=None,
        actions=None,
        pose_var=None,
    ) -> "Episode":
        """Build a T=1 Episode, e.g. for a one-shot inference state.

        ``poses`` may be shape (N, 7) (a T axis is added) or (1, N, 7).
        """
        poses = _f32(poses)
        if poses.ndim == 2:
            poses = poses[None, ...]

        def _add_t(value):
            if value is None:
                return None
            value = _f32(value)
            return value[None, ...] if value.ndim == 2 else value

        series = StateSeries(
            times=np.array([time], dtype=np.float32),
            poses=poses,
            lin_vel=_add_t(lin_vel),
            ang_vel=_add_t(ang_vel),
            actions=(_f32(actions)[None, ...] if actions is not None and _f32(actions).ndim == 1 else actions),
            pose_var=_add_t(pose_var),
        )
        return cls(scene=scene, series=series)
