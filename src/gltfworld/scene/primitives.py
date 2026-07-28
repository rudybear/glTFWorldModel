"""trimesh-based primitive mesh generation.

Used ONLY to generate geometry (uv_sphere/box/cylinder/plane); trimesh is
never used to read or write glTF/GLB itself -- that's pygltflib's job (see
``gltfworld.gltf.accessors`` and ``gltfworld.scene.convert``).

All shapes are centered at the origin, Y-up, matching glTF's coordinate
convention and the ``KHR_implicit_shapes`` shape definitions gltfworld pairs
them with. ``size`` follows the same per-shape convention as
``gltfworld.scene.scene.ObjectSpec.size``:

- sphere: ``[r, r, r]``
- box: half-extents ``[hx, hy, hz]``
- cylinder: ``[r, half_height, r]``, symmetric about the local **Y** axis
  (radius in the X/Z plane, half-height along Y) -- matching both
  ``ObjectSpec``'s documented convention and the vendored
  ``KHR_implicit_shapes`` cylinder schema's own text ("centered along the Y
  axis"). ``trimesh.creation.cylinder`` generates a mesh symmetric about
  local **Z** by default; :func:`mesh_for` rotates that output -90 degrees
  about local X (a proper rotation, applied to vertices *and* normals)
  before returning it, so the mesh this module hands back is genuinely
  Y-symmetric, not just documented as such (see the V3.1 report / DESIGN.md
  "Cylinder axis convention" for the interop bug this fixes: a
  spec-conformant external ``KHR_implicit_shapes`` reader reconstructs the
  collider centered on Y, so the visual mesh must actually agree).
"""

from __future__ import annotations

import numpy as np
import trimesh

_SPHERE_COUNT = (16, 16)  # -> ~16 latitude rings x 32 longitude segments
_CYLINDER_SECTIONS = 32

Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]


def _vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Average face normals per vertex without pulling in scipy.

    ``trimesh.Trimesh.vertex_normals`` uses a sparse-matrix weighted average
    that falls back noisily to a dense path when scipy isn't installed
    (scipy is an ML-extra dependency, not a core one here). A plain
    unweighted average is fine for the primitive shapes gltfworld generates.
    """
    face_normals = mesh.face_normals
    vertex_normals = np.zeros_like(mesh.vertices, dtype=np.float64)
    counts = np.zeros(len(mesh.vertices), dtype=np.float64)
    for face, normal in zip(mesh.faces, face_normals):
        for vertex_index in face:
            vertex_normals[vertex_index] += normal
            counts[vertex_index] += 1
    counts[counts == 0] = 1.0
    vertex_normals /= counts[:, None]
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vertex_normals / norms


def _from_trimesh(mesh: trimesh.Trimesh) -> Mesh:
    positions = np.asarray(mesh.vertices, dtype=np.float32)
    normals = _vertex_normals(mesh).astype(np.float32)
    indices = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
    return positions, normals, indices


def mesh_for(shape: str, size) -> Mesh:
    """Generate a (positions, normals, indices) triangle mesh for ``shape``.

    Returns
    -------
    positions : float32 (V, 3)
    normals : float32 (V, 3)
    indices : uint32 (F * 3,)
    """
    size = np.asarray(size, dtype=np.float64)

    if shape == "sphere":
        radius = float(size[0])
        mesh = trimesh.creation.uv_sphere(radius=radius, count=list(_SPHERE_COUNT))
    elif shape == "box":
        extents = (size * 2.0).tolist()
        mesh = trimesh.creation.box(extents=extents)
    elif shape == "cylinder":
        radius = float(size[0])
        height = float(size[1] * 2.0)
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=_CYLINDER_SECTIONS)
        # trimesh's cylinder is symmetric about local Z by default; rotate
        # -90 degrees about local X so it's symmetric about local Y instead,
        # matching ObjectSpec's documented convention and the
        # KHR_implicit_shapes cylinder schema ("centered along the Y axis").
        # A proper rotation (orthonormal, det +1) transforms vertices and
        # (via trimesh's own apply_transform) face/vertex normals correctly
        # together -- see module docstring and tests/test_cylinder_axis.py.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0]))
    elif shape == "plane":
        # No native "plane" primitive is used on the wire (ObjectSpec only
        # allows sphere/box/cylinder); a thin box stands in for a finite
        # ground plane when one is wanted, per the milestone spec.
        half_width, thickness, half_depth = size
        extents = [half_width * 2.0, max(thickness, 1e-4) * 2.0, half_depth * 2.0]
        mesh = trimesh.creation.box(extents=extents)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    return _from_trimesh(mesh)
