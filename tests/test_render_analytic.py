"""Analytic correctness tests for `gltfworld.render.renderer.EpisodeRenderer`.

These build small, hand-computed scenes directly as `Episode` objects (no
GLB round trip needed -- `EpisodeRenderer.load` takes an `Episode`) so the
expected pixel-space geometry can be derived in closed form and compared to
the actual render.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_sample_episode

from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.scene import CameraSpec, LightSpec, ObjectSpec, SceneState

pytestmark = pytest.mark.gpu

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
SIZE = 256


def _camera(distance: float, yfov: float = 0.8) -> CameraSpec:
    """Camera on the +Z axis at ``distance``, looking at the origin.

    Identity rotation means the camera looks down local -Z (glTF/OpenGL
    convention); placed at +Z it looks straight back at the origin.
    """
    return CameraSpec(
        position=np.array([0.0, 0.0, distance], dtype=np.float32),
        rotation=_IDENTITY_QUAT.copy(),
        yfov=yfov,
        znear=0.01,
        zfar=1000.0,
        aspect=1.0,
    )


def _light() -> LightSpec:
    return LightSpec(
        type="directional",
        color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        intensity=3.0,
        rotation=_IDENTITY_QUAT.copy(),
    )


def _object(object_id: int, shape: str, size, position, *, color=(0.8, 0.2, 0.2, 1.0)) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        shape=shape,
        size=np.array(size, dtype=np.float32),
        color=np.array(color, dtype=np.float32),
        roughness=0.5,
        metallic=0.0,
        mass=1.0,
        friction=0.5,
        restitution=0.3,
        is_static=True,
        category="test",
    ), position


def _episode(camera: CameraSpec, objects_and_positions: list[tuple[ObjectSpec, tuple]]) -> Episode:
    objects = [obj for obj, _ in objects_and_positions]
    scene = SceneState(
        objects=objects,
        camera=camera,
        lights=[_light()],
        gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        dt=1.0 / 60.0,
        seed=0,
    )
    n = len(objects)
    poses = np.zeros((1, n, 7), dtype=np.float32)
    for i, (_, position) in enumerate(objects_and_positions):
        poses[0, i, 0:3] = position
        poses[0, i, 3:7] = _IDENTITY_QUAT
    series = StateSeries(times=np.array([0.0], dtype=np.float32), poses=poses)
    return Episode(scene=scene, series=series)


def _expected_pixel_radius(yfov: float, height: int, distance: float, radius: float) -> float:
    """Closed-form projected pixel radius of a sphere of ``radius`` centered
    on the camera's principal axis at ``distance``.

    The sphere's silhouette edge is tangent to the sphere from the camera,
    at angular radius ``theta = asin(radius / distance)`` off the principal
    axis. Perspective projection maps a ray at angle ``theta`` off-axis to
    normalized device offset ``tan(theta) / tan(yfov / 2)`` (the frustum's
    vertical half-angle ``yfov / 2`` maps to the +/-1 NDC edge), which then
    scales to ``height / 2`` pixels.
    """
    theta = math.asin(radius / distance)
    ndc_offset = math.tan(theta) / math.tan(yfov / 2.0)
    return ndc_offset * (height / 2.0)


def test_sphere_silhouette_area_matches_closed_form(episode_renderer):
    distance = 6.0
    radius = 1.0
    yfov = 0.8
    obj, pos = _object(1, "sphere", [radius, radius, radius], (0.0, 0.0, 0.0))
    episode = _episode(_camera(distance, yfov), [(obj, pos)])

    episode_renderer.load(episode)
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()

    expected_pixel_radius = _expected_pixel_radius(yfov, SIZE, distance, radius)
    expected_area = math.pi * expected_pixel_radius**2

    actual_area = int(np.count_nonzero(frame.seg == 1))

    rel_error = abs(actual_area - expected_area) / expected_area
    assert rel_error <= 0.03, (
        f"silhouette area off by {rel_error:.4%}: actual={actual_area} expected={expected_area:.1f}"
    )


def test_sphere_center_depth_matches_closed_form(episode_renderer):
    distance = 6.0
    radius = 1.0
    obj, pos = _object(1, "sphere", [radius, radius, radius], (0.0, 0.0, 0.0))
    episode = _episode(_camera(distance), [(obj, pos)])

    episode_renderer.load(episode)
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()

    center = SIZE // 2
    expected_depth = distance - radius
    actual_depth = float(frame.depth[center, center])

    assert actual_depth > 0.0, "center pixel should hit the sphere, not background"
    rel_error = abs(actual_depth - expected_depth) / expected_depth
    assert rel_error <= 0.01, f"center depth off by {rel_error:.4%}: actual={actual_depth} expected={expected_depth}"


def _ndc_pixel_offset(yfov: float, size: int, world_offset: float, depth: float) -> float:
    """Pixel offset from image center for a point ``world_offset`` off the
    principal axis at eye-space ``depth`` (same tan-based mapping as
    ``_expected_pixel_radius``, but for a point at a known world offset
    rather than a sphere's tangent-angle silhouette)."""
    return (size / 2.0) * (world_offset / depth) / math.tan(yfov / 2.0)


def test_box_occlusion_seg_and_depth_ordering(episode_renderer):
    # Front box: closer to the camera, smaller -- fully covers the principal
    # axis but not the whole back box. Back box: further away, larger, so it
    # peeks out around the front box's edges (proving occlusion, not mere
    # absence, and giving an unambiguous depth-ordering check at the center).
    camera_z = 6.0
    yfov = 0.8
    front_half = 0.4
    back_half = 1.5
    front_obj, front_pos = _object(
        1, "box", [front_half, front_half, front_half], (0.0, 0.0, 2.0), color=(0.9, 0.1, 0.1, 1.0)
    )
    back_obj, back_pos = _object(
        2, "box", [back_half, back_half, back_half], (0.0, 0.0, -2.0), color=(0.1, 0.1, 0.9, 1.0)
    )
    episode = _episode(_camera(camera_z, yfov), [(front_obj, front_pos), (back_obj, back_pos)])

    episode_renderer.load(episode)
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()

    center = SIZE // 2
    assert frame.seg[center, center] == 1, "center pixel should show the front (occluding) object"

    expected_depth = camera_z - front_pos[2] - front_half  # front box's near face
    actual_depth = float(frame.depth[center, center])
    rel_error = abs(actual_depth - expected_depth) / expected_depth
    assert rel_error <= 0.01, f"occluded depth off by {rel_error:.4%}"

    front_depth = camera_z - front_pos[2]
    back_depth = camera_z - back_pos[2]
    front_px = _ndc_pixel_offset(yfov, SIZE, front_half, front_depth)
    back_px = _ndc_pixel_offset(yfov, SIZE, back_half, back_depth)
    assert back_px > front_px + 5, "test geometry must make the back box visibly wider on-screen"
    peek_col = center + int((front_px + back_px) / 2)

    assert frame.seg[center, peek_col] == 2, (
        "back object should peek out around the front object's silhouette, proving occlusion"
    )


def test_seg_exactness_only_known_object_ids_or_background(episode_renderer):
    episode = make_sample_episode()
    valid_ids = {obj.object_id for obj in episode.scene.objects}

    episode_renderer.load(episode)
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()
    present_ids = set(np.unique(frame.seg).tolist())
    assert present_ids <= (valid_ids | {0}), (
        f"seg contains ids not in the episode (or background): {present_ids - (valid_ids | {0})}"
    )


def test_determinism_same_frame_renders_bitwise_identical(episode_renderer):
    episode = make_sample_episode()
    episode_renderer.load(episode)
    episode_renderer.set_frame(5)
    a = episode_renderer.render()
    episode_renderer.set_frame(5)
    b = episode_renderer.render()

    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.seg, b.seg)
    # Bit-for-bit float32 comparison (uint32 view), not just np.allclose.
    assert np.array_equal(a.depth.view(np.uint32), b.depth.view(np.uint32))
