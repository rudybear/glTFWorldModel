"""Scene-description dataclasses: the static, per-episode content.

These describe *what* is in a scene (objects, camera, lights, physics
constants) but not *how it moves over time* -- that's ``StateSeries`` in
``gltfworld.scene.episode``. Everything numeric is stored as float32 numpy
arrays so an encode -> glTF float32 accessor -> decode round trip is bitwise
exact.

``ObjectSpec.size`` is (3,) float32 with a per-shape convention (also used
by ``gltfworld.scene.primitives.mesh_for`` and the ``KHR_implicit_shapes``
codec in ``gltfworld.ext.khr_physics``):

- sphere: ``[r, r, r]`` (isotropic; only ``size[0]`` is actually read)
- box: half-extents ``[hx, hy, hz]``
- cylinder: ``[r, half_height, r]`` -- both radius slots must match.
  ``KHR_implicit_shapes``' cylinder supports independent top/bottom radii
  (a frustum); ``ObjectSpec`` only ever represents a true cylinder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SHAPES = ("sphere", "box", "cylinder")
LIGHT_TYPES = ("directional", "point")


def _f32(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def _f32_scalar(value: Any) -> float:
    """Snap a scalar to the float32 grid, stored as a Python float.

    Scalar physics/material properties (roughness, mass, friction, ...)
    aren't carried in binary accessors -- they're embedded directly as JSON
    numbers in glTF materials/extensions, which have full float64 range.
    Snapping to float32 here keeps them consistent with the rest of the
    (float32) dataclasses and means an encode -> JSON -> decode round trip
    (which preserves float64 exactly) reproduces the exact original value.
    """
    return float(np.float32(value))


@dataclass
class ObjectSpec:
    """A single rigid-body object in a scene.

    ``object_id`` is stable across an episode (and across encode/decode) so
    downstream consumers (RWM extras, KHR physics nodes, animation targets)
    can all key off it.
    """

    object_id: int
    shape: str
    size: np.ndarray  # (3,) float32; see module docstring for per-shape meaning
    color: np.ndarray  # (4,) float32 rgba
    roughness: float
    metallic: float
    mass: float
    friction: float
    restitution: float
    is_static: bool
    category: str
    parts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}, expected one of {SHAPES}")
        self.object_id = int(self.object_id)
        self.size = _f32(self.size, (3,))
        self.color = _f32(self.color, (4,))
        self.roughness = _f32_scalar(self.roughness)
        self.metallic = _f32_scalar(self.metallic)
        self.mass = _f32_scalar(self.mass)
        self.friction = _f32_scalar(self.friction)
        self.restitution = _f32_scalar(self.restitution)
        self.is_static = bool(self.is_static)
        self.category = str(self.category)


@dataclass
class CameraSpec:
    """A single camera; gltfworld scenes carry exactly one."""

    position: np.ndarray  # (3,) float32
    rotation: np.ndarray  # (4,) float32 quat xyzw
    yfov: float
    znear: float
    zfar: float
    aspect: float

    def __post_init__(self) -> None:
        self.position = _f32(self.position, (3,))
        self.rotation = _f32(self.rotation, (4,))
        self.yfov = _f32_scalar(self.yfov)
        self.znear = _f32_scalar(self.znear)
        self.zfar = _f32_scalar(self.zfar)
        self.aspect = _f32_scalar(self.aspect)


@dataclass
class LightSpec:
    """A single light: directional lights use rotation, point lights use position."""

    type: str
    color: np.ndarray  # (3,) float32
    intensity: float
    rotation: np.ndarray | None = None  # (4,) float32 quat xyzw, directional
    position: np.ndarray | None = None  # (3,) float32, point

    def __post_init__(self) -> None:
        if self.type not in LIGHT_TYPES:
            raise ValueError(f"unknown light type {self.type!r}, expected one of {LIGHT_TYPES}")
        self.color = _f32(self.color, (3,))
        self.intensity = _f32_scalar(self.intensity)
        if self.rotation is not None:
            self.rotation = _f32(self.rotation, (4,))
        if self.position is not None:
            self.position = _f32(self.position, (3,))
        if self.type == "directional" and self.rotation is None:
            raise ValueError("directional light requires rotation")
        if self.type == "point" and self.position is None:
            raise ValueError("point light requires position")


@dataclass
class SceneState:
    """The static content of one episode: objects, camera, lights, physics constants."""

    objects: list[ObjectSpec]
    camera: CameraSpec
    lights: list[LightSpec]
    gravity: np.ndarray  # (3,) float32
    dt: float
    seed: int
    scene_version: str = "wm-scenes-v1"

    def __post_init__(self) -> None:
        self.gravity = _f32(self.gravity, (3,))
        self.dt = _f32_scalar(self.dt)
        self.seed = int(self.seed)
        self.scene_version = str(self.scene_version)
