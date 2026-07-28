"""Cylinder Y-axis convention: the V3.1 interop-defect fix (see DESIGN.md
"Cylinder axis convention" and the V3.1 report).

Three independent proofs that "cylinder is symmetric about local **Y**" now
holds everywhere gltfworld touches a cylinder, not just in one place that
happens to agree with another equally-wrong place:

1. :func:`test_mesh_is_y_symmetric` -- the raw trimesh-generated mesh's
   vertex bounds directly (radius in X/Z, half-height along Y).
2. :func:`test_khr_spec_axis_matches_actual_mesh_axis_under_node_transform`
   -- decode a real GLB's mesh POSITION accessor (not gltfworld's own
   source code) to independently re-derive the mesh's actual local
   symmetry axis, and confirm it agrees with the axis a spec-conformant
   ``KHR_implicit_shapes`` reader would reconstruct ("centered along Y"),
   under the *same* node rotation -- this is exactly the check an external
   engine implicitly performs, and exactly what the defect broke.
3. :func:`test_cylinder_on_its_side_rests_at_radius_height` -- a physical
   proof via a real MuJoCo simulation: a cylinder dropped lying on its side
   must come to rest with its center a *radius* above the ground (resting
   on its curved side), not a *half-height* above it (which is what a
   Z-symmetric physics geom, or a support-offset helper still assuming
   Z is the height axis, would produce).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_sample_episode

from gltfworld.gltf.accessors import read_accessor
from gltfworld.scene.convert import episode_to_gltf
from gltfworld.scene.primitives import mesh_for
from gltfworld.scene.scene import CameraSpec, LightSpec, ObjectSpec, SceneState

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --- (a) mesh vertex-bounds prove Y is the symmetry axis --------------------


def test_mesh_is_y_symmetric():
    radius, half_height = 0.15, 0.4
    positions, normals, _indices = mesh_for("cylinder", [radius, half_height, radius])

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = (maxs - mins) / 2.0  # half-extent per axis

    # Half-height along Y, radius in X/Z -- and X/Z extents must actually
    # match each other (a real radius), not just both be smaller than Y.
    assert extent[1] == pytest.approx(half_height, abs=1e-5), f"Y half-extent should be half_height, got {extent}"
    assert extent[0] == pytest.approx(radius, abs=1e-5), f"X half-extent should be radius, got {extent}"
    assert extent[2] == pytest.approx(radius, abs=1e-5), f"Z half-extent should be radius, got {extent}"
    assert extent[1] > extent[0] and extent[1] > extent[2], "Y must be the elongated (height) axis"

    # Also centered at the origin (symmetric, not just bounded) on every axis.
    center = (maxs + mins) / 2.0
    np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-5)

    # Normals must actually have been carried along with the vertex rotation
    # (not left pointing the old, Z-symmetric way): every vertex on this
    # simple two-ring cylinder mesh sits on a rim shared between a cap and
    # the wall, so its (unweighted-averaged) normal has a Y component whose
    # *sign* must match the vertex's own Y sign (top-rim vertices point
    # generally "up-and-out", bottom-rim vertices "down-and-out") -- this
    # would flip to correlating with Z instead of Y if the mesh (and its
    # normals) were still Z-symmetric.
    assert np.all(np.linalg.norm(normals, axis=1) > 0.99), "normals must be unit length"
    nonzero_y = np.abs(positions[:, 1]) > 1e-6
    assert np.all(np.sign(normals[nonzero_y, 1]) == np.sign(positions[nonzero_y, 1])), (
        "normal Y-component sign must match vertex Y-sign -- normals didn't follow the Y-axis fix"
    )


# --- (b) spec-reconstructed collider axis vs. the actual decoded mesh axis --


def _decoded_local_symmetry_axis(gltf, node_index: int) -> np.ndarray:
    """Independently re-derive the local axis a mesh is elongated along,
    from the *decoded* POSITION accessor of the node's mesh (not from
    gltfworld's own primitives.py source) -- the axis with the largest
    vertex-bound half-extent."""
    node = gltf.nodes[node_index]
    mesh = gltf.meshes[node.mesh]
    pos_accessor_index = mesh.primitives[0].attributes.POSITION
    positions = read_accessor(gltf, pos_accessor_index)
    extent = (positions.max(axis=0) - positions.min(axis=0)) / 2.0
    axis = np.zeros(3)
    axis[int(np.argmax(extent))] = 1.0
    return axis


def test_khr_spec_axis_matches_actual_mesh_axis_under_node_transform():
    episode = make_sample_episode()
    cylinder_obj = next(obj for obj in episode.scene.objects if obj.shape == "cylinder")
    cylinder_index = episode.scene.objects.index(cylinder_obj)

    # This episode's cylinder is deliberately given a non-identity, non-90-
    # degree-multiple-preserving orientation is not required -- even the
    # fixed conftest "lying on its side" quaternion (90 degrees about world
    # Z) is enough: it does not send the Y axis onto the Z axis, so a
    # regression back to Z-symmetric mesh geometry would clearly fail the
    # alignment check below rather than passing by coincidence.
    node_quat = episode.series.poses[0, cylinder_index, 3:7]

    gltf = episode_to_gltf(episode)
    node_index = next(i for i, n in enumerate(gltf.nodes) if n.name == f"obj_{cylinder_obj.object_id}")

    # Sanity: the node's own rotation (what a viewer/engine would apply)
    # really is what we think it is.
    np.testing.assert_allclose(gltf.nodes[node_index].rotation, node_quat, atol=1e-6)

    decoded_local_axis = _decoded_local_symmetry_axis(gltf, node_index)
    # The actual encoded mesh must be Y-symmetric (this is the fix under
    # test -- if this fails, everything downstream is moot).
    np.testing.assert_allclose(decoded_local_axis, [0.0, 1.0, 0.0], atol=1e-9)

    khr_spec_local_axis = np.array([0.0, 1.0, 0.0])  # KHR_implicit_shapes: "centered along the Y axis"

    R = _quat_to_matrix_xyzw(node_quat)
    world_axis_from_mesh = R @ decoded_local_axis
    world_axis_from_khr_spec = R @ khr_spec_local_axis

    cos_angle = float(np.dot(world_axis_from_mesh, world_axis_from_khr_spec))
    assert abs(cos_angle) > 0.999, (
        f"spec-reconstructed collider axis {world_axis_from_khr_spec} disagrees with the "
        f"actual mesh's symmetry axis {world_axis_from_mesh} under the shared node rotation "
        f"(cos angle={cos_angle:.4f}) -- exactly the interop defect this fix addresses"
    )

    # And, since they're the same local axis to begin with here, they
    # should in fact be numerically identical (not just "aligned"), which
    # would not hold if the mesh's real local axis were still Z.
    np.testing.assert_allclose(world_axis_from_mesh, world_axis_from_khr_spec, atol=1e-6)


# --- (c) physical proof: resting height is radius, not half-height ----------


def _minimal_scene(objects: list[ObjectSpec]) -> SceneState:
    camera = CameraSpec(
        position=np.array([0.0, 1.0, 3.0], dtype=np.float32),
        rotation=_IDENTITY_QUAT.copy(),
        yfov=0.8,
        znear=0.05,
        zfar=100.0,
        aspect=1.0,
    )
    lights = [
        LightSpec(
            type="directional",
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=3.0,
            rotation=_IDENTITY_QUAT.copy(),
        )
    ]
    return SceneState(
        objects=objects,
        camera=camera,
        lights=lights,
        gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        dt=1.0 / 30.0,
        seed=0,
    )


def test_cylinder_on_its_side_rests_at_radius_height():
    pytest.importorskip("mujoco")
    from gltfworld.datagen.mujoco_env import simulate
    from gltfworld.datagen.sample import object_support_offset

    radius, half_height = 0.12, 0.35  # deliberately far apart: 0.12 vs 0.35

    ground_half_extents = np.array([2.0, 0.1, 2.0], dtype=np.float32)
    ground = ObjectSpec(
        object_id=0,
        shape="box",
        size=ground_half_extents,
        color=np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1000.0,
        friction=0.9,
        restitution=0.05,
        is_static=True,
        category="ground",
    )
    cylinder = ObjectSpec(
        object_id=1,
        shape="cylinder",
        size=np.array([radius, half_height, radius], dtype=np.float32),
        color=np.array([0.8, 0.2, 0.2, 1.0], dtype=np.float32),
        roughness=0.5,
        metallic=0.0,
        mass=1.0,
        friction=0.9,
        restitution=0.05,
        is_static=False,
        category="cylinder",
    )
    scene = _minimal_scene([ground, cylinder])

    ground_top_y = 0.0
    # 90 degrees about world X: the cylinder's local Y (its height axis)
    # ends up horizontal (along world Z) -- lying on its side.
    on_side_quat = np.array([math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)], dtype=np.float32)
    initial_poses = np.zeros((2, 7), dtype=np.float32)
    initial_poses[0, 1] = ground_top_y - float(ground_half_extents[1])
    initial_poses[0, 3:7] = _IDENTITY_QUAT
    initial_poses[1, 1] = ground_top_y + radius + 0.25  # dropped from just above resting height
    initial_poses[1, 3:7] = on_side_quat

    series = simulate(scene, initial_poses, T=120, record_hz=60.0)

    final_pose = series.poses[-1, 1]
    final_y = float(final_pose[1])
    final_quat = final_pose[3:7]

    # It really did fall and settle (not stuck exactly at the drop height).
    assert final_y < initial_poses[1, 1] - 0.05

    offset = object_support_offset(cylinder, final_quat, np.array([0.0, -1.0, 0.0]))
    assert offset == pytest.approx(radius, abs=0.02), (
        f"resting support offset {offset:.4f} should be ~radius ({radius}) for a cylinder "
        "lying on its side, not ~half_height"
    )
    assert abs(offset - half_height) > 0.1, "support offset must clearly not be the half-height instead"

    bottom = final_y - offset
    assert bottom == pytest.approx(ground_top_y, abs=0.02), (
        f"cylinder's true bottom ({bottom:.4f}) should have settled at the ground top "
        f"({ground_top_y}); resting center height was {final_y:.4f}, offset {offset:.4f}"
    )

    # And, directly, the *center* height itself should be close to
    # ground + radius (not ground + half_height) -- the headline physical
    # claim of this test.
    assert final_y == pytest.approx(ground_top_y + radius, abs=0.02)
    assert abs(final_y - (ground_top_y + half_height)) > 0.1
