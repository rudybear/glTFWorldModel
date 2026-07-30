"""``gltfworld.physion.convert`` against real Physion Collide HDF5 trials.

Like ``tests/test_physion_ingest.py``, this skips cleanly (with a clear
reason) when the real, gitignored HDF5 tier isn't present on this machine --
CI won't have it; it's real external benchmark data, not something CI
generates. Unit tests that don't need real HDF5 data (the bounding-shape
classifier, the normal computation) always run.
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pygltflib
import pytest

from gltfworld.physion import convert as pc
from gltfworld.scene.convert import episode_from_gltf

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_DIR = REPO_ROOT / "data" / "external" / "physion" / "hdf5" / "extracted" / "Collide" / "hdf5s"


def _require_real_hdf5() -> list[Path]:
    if not HDF5_DIR.exists():
        pytest.skip(
            f"{HDF5_DIR} not present -- download+extract the Collide test HDF5 tier "
            "there first (see docs/PHYSION.md); this is real external benchmark data, "
            "not something CI generates or ships"
        )
    files = sorted(HDF5_DIR.glob("*.hdf5"))
    if not files:
        pytest.skip(f"{HDF5_DIR} present but empty")
    return files


def _network_and_binary_unavailable() -> bool:
    from gltfworld.cli import ensure_validator_binary

    try:
        ensure_validator_binary()
        return False
    except Exception:
        pass
    try:
        socket.create_connection(("github.com", 443), timeout=3).close()
        return False
    except OSError:
        return True


# Deterministic indices into the sorted 150-file corpus, spread across its
# whole span (V8.1 fix -- the original V8 test hardcoded `files[:3]`, which
# an independent verifier flagged: 3 trials all drawn from the front of the
# sorted list never exercised the zero-length-normal cancellation defect
# that 12/150 real trials hit, see docs/VERIFICATION.md's V8.1 section).
_SAMPLE_INDICES = [0, 29, 59, 89, 119, 149]

# One of the 12 trials the V8.1 validator sweep found failing with
# ACCESSOR_VECTOR3_NON_UNIT pre-fix (index 74 in the sorted corpus) --
# pinned explicitly (on top of the spread above) so this regression can
# never silently pass by missing the file that actually exercises it.
_PINNED_FAILING_TRIAL = "pilot_it2_collision_simple_tdw_2_dis_2_occ_0002"


@pytest.fixture(scope="module")
def sampled_trials():
    files = _require_real_hdf5()
    selected = {files[i] for i in _SAMPLE_INDICES}
    pinned = [fp for fp in files if fp.stem == _PINNED_FAILING_TRIAL]
    assert pinned, f"{_PINNED_FAILING_TRIAL} not found in {HDF5_DIR}"
    selected.add(pinned[0])
    return [pc.load_trial(fp) for fp in sorted(selected)]


# --- unit tests, no real data needed --------------------------------------------


def test_classify_bounding_shape_sphere_isotropic():
    # a rough icosahedron-like point cloud, isotropic AABB -> "sphere"
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(200, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vertices = (dirs * 0.5).astype(np.float32)
    shape, size, center = pc.classify_bounding_shape(vertices)
    assert shape == "sphere"
    assert np.allclose(size, size[0], atol=1e-3)
    assert np.allclose(center, 0.0, atol=1e-2)


def test_classify_bounding_shape_box_anisotropic():
    vertices = np.array(
        [[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 2.0, -0.5], [-0.5, 2.0, 0.5]], dtype=np.float32
    )
    shape, size, center = pc.classify_bounding_shape(vertices)
    assert shape == "box"
    assert size[1] > size[0]  # tall box, y half-extent (1.0) >> x half-extent (0.5)
    assert np.allclose(center, [0.0, 1.0, 0.0], atol=1e-6)


def test_compute_vertex_normals_simple_triangle():
    # a single triangle in the XY plane, CCW when viewed from +Z -> normal +Z
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    normals = pc.compute_vertex_normals(vertices, faces)
    assert normals.shape == (3, 3)
    for n in normals:
        assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-5)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-5)


def test_compute_vertex_normals_cancellation_falls_back_to_first_face():
    # V8.1 regression test: a synthetic mesh where vertex 0's area-weighted
    # face-normal sum cancels to *exactly* zero (two equal-area faces with
    # exactly opposite normals sharing vertex 0), the real defect an
    # independent verifier found producing ACCESSOR_VECTOR3_NON_UNIT on
    # 12/150 real Collide GLBs (see docs/VERIFICATION.md's V8.1 section).
    # v0 is the shared/degenerate vertex; v1/v2 belong only to face A
    # (normal +Z), v3/v4 belong only to face B (normal -Z, equal area).
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # v0: shared, degenerate after summation
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2
            [-1.0, 0.0, 0.0],  # v3
            [0.0, -1.0, 0.0],  # v4
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 4, 3]], dtype=np.uint32)  # face A: +Z, face B: -Z

    # sanity: the two face normals really are equal-magnitude and opposite,
    # so v0's naive area-weighted sum is exactly zero (the actual defect).
    v0, v1, v2, v3, v4 = vertices.astype(np.float64)
    normal_a = np.cross(v1 - v0, v2 - v0)
    normal_b = np.cross(v4 - v0, v3 - v0)
    assert np.allclose(normal_a + normal_b, 0.0)
    assert np.linalg.norm(normal_a) > 0

    normals = pc.compute_vertex_normals(vertices, faces)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "every fallback normal must still be unit length"

    # v0 falls back to face A's normal (first adjacent face in face-array order)
    assert np.allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-5)
    # non-degenerate vertices are unaffected by the fallback path
    assert np.allclose(normals[1], [0.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(normals[3], [0.0, 0.0, -1.0], atol=1e-5)


def test_compute_vertex_normals_fully_degenerate_falls_back_to_up():
    # if even the first adjacent face is itself degenerate (zero area, e.g.
    # three coincident points), fall back to a fixed +Y rather than
    # producing (or propagating) a zero-length normal.
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    normals = pc.compute_vertex_normals(vertices, faces)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    for n in normals:
        assert np.allclose(n, [0.0, 1.0, 0.0], atol=1e-5)


# --- real-data tests -------------------------------------------------------------


def test_load_trial_basic_shape(sampled_trials):
    for trial in sampled_trials:
        assert trial.scenario == "Collide"
        assert len(trial.objects) >= 3
        assert trial.times.shape[0] == len(trial.contact_per_frame)
        assert trial.label == bool(np.any(trial.contact_per_frame))
        object_ids = {obj.object_id for obj in trial.objects}
        assert trial.target_id in object_ids
        assert trial.zone_id in object_ids
        assert trial.probe_id in object_ids
        roles = {obj.object_id: obj.role for obj in trial.objects}
        assert roles[trial.target_id] == pc.ROLE_AGENT
        assert roles[trial.zone_id] == pc.ROLE_PATIENT
        assert roles[trial.probe_id] == pc.ROLE_PROBE


def test_pose_animation_frame_count_matches_hdf5(sampled_trials, tmp_path):
    for trial in sampled_trials:
        glb_path = tmp_path / f"{trial.trial_id}.glb"
        pc.save_trial_gltf(trial, glb_path)
        gltf = pygltflib.GLTF2.load(str(glb_path))
        episode = episode_from_gltf(gltf)
        assert episode.series.num_frames == len(trial.times)
        assert episode.series.num_frames == len(trial.contact_per_frame)


def test_round_trip_poses_and_velocities(sampled_trials, tmp_path):
    for trial in sampled_trials:
        glb_path = tmp_path / f"{trial.trial_id}.glb"
        pc.save_trial_gltf(trial, glb_path)
        gltf = pygltflib.GLTF2.load(str(glb_path))
        episode = episode_from_gltf(gltf)

        for i, obj in enumerate(trial.objects):
            assert np.allclose(episode.series.poses[:, i, 0:3], obj.positions, atol=1e-5)
            # quaternion hemisphere may flip sign on decode (contract-level
            # convention, not a bug -- see gltfworld.scene.contract docstring)
            rot_decoded = episode.series.poses[:, i, 3:7]
            same = np.allclose(rot_decoded, obj.rotations, atol=1e-5)
            flipped = np.allclose(rot_decoded, -obj.rotations, atol=1e-5)
            assert same or flipped
            assert np.allclose(episode.series.lin_vel[:, i, :], obj.lin_vel, atol=1e-4)
            assert np.allclose(episode.series.ang_vel[:, i, :], obj.ang_vel, atol=1e-4)

            decoded_obj = episode.scene.objects[i]
            assert decoded_obj.object_id == obj.object_id
            assert decoded_obj.category == obj.model_name
            assert decoded_obj.is_static == obj.is_static
            assert decoded_obj.parts["physion_role"] == obj.role


def test_real_mesh_round_trips(sampled_trials, tmp_path):
    for trial in sampled_trials:
        glb_path = tmp_path / f"{trial.trial_id}.glb"
        pc.save_trial_gltf(trial, glb_path)
        gltf = pygltflib.GLTF2.load(str(glb_path))
        node_indices = pc.read_object_node_indices(gltf)
        assert len(node_indices) == len(trial.objects)

        for obj, node_index in zip(trial.objects, node_indices):
            positions, normals, faces = pc.read_mesh_accessors(gltf, node_index)
            assert np.allclose(positions, obj.vertices)
            assert np.array_equal(faces, obj.faces)
            assert positions.shape[0] == obj.vertices.shape[0]
            norms = np.linalg.norm(normals, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3)


@pytest.mark.skipif(_network_and_binary_unavailable(), reason="no cached glTF-Validator binary and no network to fetch it")
def test_converted_trials_validate_clean(sampled_trials, tmp_path):
    from gltfworld.cli import run_validator

    for trial in sampled_trials:
        glb_path = tmp_path / f"{trial.trial_id}.glb"
        pc.save_trial_gltf(trial, glb_path)

        report = run_validator(str(glb_path))
        issues = report["issues"]
        assert issues["numErrors"] == 0, issues["messages"]
        warning_codes = {m["code"] for m in issues["messages"] if m["severity"] == 1}
        assert warning_codes <= {"UNSUPPORTED_EXTENSION"}, issues["messages"]
