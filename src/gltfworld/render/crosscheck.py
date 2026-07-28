"""MuJoCo cross-render oracle: an independent check that `EpisodeRenderer`'s
geometry (not lighting/color -- MuJoCo's flat shading looks nothing like
pyrender's PBR) matches what a completely different renderer thinks the
scene looks like, for the same episode's frame 0.

Requires the ``sim`` extra (``mujoco>=3.1``); import lazily / guard with
``pytest.importorskip("mujoco")`` at call sites that must run without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MuJoCo picks its own GL backend independent of PyOpenGL's
# PYOPENGL_PLATFORM (see gltfworld.render.renderer); left unset it tries
# GLX against the X server, which collides with the process's existing EGL
# context (from gltfworld's own EpisodeRenderer) with a GLX BadAccess X11
# error. Force MuJoCo's own EGL backend too, unless the caller already set
# something. Must happen before `import mujoco`.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from gltfworld.scene.episode import Episode

# --- coordinate conversion ----------------------------------------------------

# +90 degree rotation about the shared X axis, as an xyzw quaternion: this is
# the fixed rotation that carries glTF's Y-up right-handed world frame onto
# MuJoCo's Z-up right-handed world frame (see `gltf_pose_to_mujoco`).
_AXIS_QUAT_XYZW = np.array(
    [np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)], dtype=np.float64
)


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (q1 applied after q2, i.e.
    the rotation ``q1 * q2`` means "first rotate by q2, then by q1")."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def gltf_pose_to_mujoco(
    position: np.ndarray, quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one (position, orientation) pose from glTF convention to
    MuJoCo convention.

    glTF: Y-up, right-handed world; cameras look down local -Z with local
    +Y up; quaternions are (x, y, z, w).
    MuJoCo: Z-up, right-handed world; cameras *also* look down local -Z
    with local +Y up (MuJoCo's camera convention already matches OpenGL/
    glTF here -- no extra camera-specific handling needed); quaternions are
    (w, x, y, z).

    The two world frames are related by a single fixed +90 degree rotation
    about the (shared) X axis: ``(x, y, z)_gltf -> (x, -z, y)_mujoco``. This
    is a proper rotation (determinant +1, not a reflection) that carries
    glTF's up axis (+Y) onto MuJoCo's up axis (+Z), so handedness/chirality
    of any geometry is preserved -- only which axis is "up" changes.
    Positions transform by that same linear map; orientations transform by
    composing the fixed axis-change rotation with the original rotation
    (``q_mujoco = q_axis_change * q_gltf``, applied in world space), since
    an object's local axes must map into the *new* world frame the same
    way the world frame itself was remapped.

    Returns
    -------
    position_mj : (3,) float64, MuJoCo (x, y, z)
    quat_wxyz_mj : (4,) float64, MuJoCo (w, x, y, z)
    """
    position = np.asarray(position, dtype=np.float64)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)

    x, y, z = position
    position_mj = np.array([x, -z, y], dtype=np.float64)

    q_mj_xyzw = _quat_mul_xyzw(_AXIS_QUAT_XYZW, quat_xyzw)
    quat_wxyz_mj = np.array(
        [q_mj_xyzw[3], q_mj_xyzw[0], q_mj_xyzw[1], q_mj_xyzw[2]], dtype=np.float64
    )
    return position_mj, quat_wxyz_mj


# --- MJCF construction ---------------------------------------------------------


def _geom_xml(obj, position_mj: np.ndarray, quat_wxyz_mj: np.ndarray) -> str:
    pos_str = " ".join(f"{v:.9g}" for v in position_mj)
    quat_str = " ".join(f"{v:.9g}" for v in quat_wxyz_mj)
    rgba_str = " ".join(f"{v:.9g}" for v in obj.color)

    if obj.shape == "sphere":
        size_str = f"{obj.size[0]:.9g}"
        geom_type = "sphere"
    elif obj.shape == "box":
        size_str = " ".join(f"{v:.9g}" for v in obj.size)  # already half-extents
        geom_type = "box"
    elif obj.shape == "cylinder":
        size_str = f"{obj.size[0]:.9g} {obj.size[1]:.9g}"  # radius, half-height
        geom_type = "cylinder"
    else:
        raise ValueError(f"unsupported shape for MJCF mirror: {obj.shape!r}")

    return (
        f'<body name="body_{obj.object_id}" pos="{pos_str}" quat="{quat_str}">'
        f'<geom name="obj_{obj.object_id}" type="{geom_type}" size="{size_str}" '
        f'rgba="{rgba_str}"/></body>'
    )


def build_mjcf(episode: Episode) -> str:
    """Build a minimal MJCF string mirroring ``episode``'s frame 0: same
    primitive shapes/sizes/poses, same camera, flat lighting.

    No joints are added (this only needs to render one static frame, never
    simulated) -- every object body is rigidly placed at its frame-0 pose
    directly under the worldbody, converted via `gltf_pose_to_mujoco`.
    """
    scene = episode.scene
    poses0 = episode.series.poses[0]

    body_xmls = []
    for i, obj in enumerate(scene.objects):
        position_mj, quat_wxyz_mj = gltf_pose_to_mujoco(poses0[i, 0:3], poses0[i, 3:7])
        body_xmls.append(_geom_xml(obj, position_mj, quat_wxyz_mj))

    cam = scene.camera
    cam_pos_mj, cam_quat_mj = gltf_pose_to_mujoco(cam.position, cam.rotation)
    cam_pos_str = " ".join(f"{v:.9g}" for v in cam_pos_mj)
    cam_quat_str = " ".join(f"{v:.9g}" for v in cam_quat_mj)
    import math

    fovy_deg = math.degrees(cam.yfov)

    return f"""<mujoco>
  <visual>
    <map znear="{cam.znear:.9g}" zfar="{cam.zfar:.9g}"/>
  </visual>
  <worldbody>
    <light diffuse="1 1 1" specular="0 0 0" pos="0 0 5" dir="0 0 -1" directional="true"/>
    <camera name="cam" pos="{cam_pos_str}" quat="{cam_quat_str}" fovy="{fovy_deg:.9g}"/>
    {"".join(body_xmls)}
  </worldbody>
</mujoco>
"""


# --- rendering -------------------------------------------------------------------


@dataclass
class MujocoFrame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, real distance in meters (large value = background)
    geom_id: np.ndarray  # (H, W) int32, -1 = background, else MuJoCo geom id
    geom_id_to_object_id: dict[int, int]


def render_mujoco_frame0(episode: Episode, width: int = 256, height: int = 256) -> MujocoFrame:
    """Render frame 0 of ``episode`` with MuJoCo, returning rgb/depth/geom-id buffers."""
    import mujoco

    mjcf = build_mjcf(episode)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")

    geom_id_to_object_id = {}
    for obj in episode.scene.objects:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{obj.object_id}")
        geom_id_to_object_id[geom_id] = obj.object_id

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam_id)
        seg = renderer.render().copy()
        geom_id = seg[..., 0]

        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_id)
        depth = renderer.render().copy()
    finally:
        renderer.close()

    return MujocoFrame(rgb=rgb, depth=depth, geom_id=geom_id, geom_id_to_object_id=geom_id_to_object_id)


# --- comparison -------------------------------------------------------------------


@dataclass
class CrosscheckResult:
    iou: float
    per_object_iou: dict[int, float]
    gltf_rgb: np.ndarray
    mujoco_rgb: np.ndarray
    gltf_mask: np.ndarray
    mujoco_mask: np.ndarray


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 1.0
    return intersection / union


def crosscheck_frame0(episode: Episode, episode_renderer, width: int = 256, height: int = 256) -> CrosscheckResult:
    """Render frame 0 with both `EpisodeRenderer` and MuJoCo and compare
    binary silhouettes (any-object vs background) plus per-object IoU where
    feasible.

    ``episode_renderer`` must already have ``width == height == (width,
    height)`` here (reuse the process's single persistent renderer -- see
    `EpisodeRenderer`'s process-lifetime constraint).
    """
    episode_renderer.load(episode)
    episode_renderer.set_frame(0)
    gltf_frame = episode_renderer.render()
    gltf_mask = gltf_frame.depth > 0.0

    mj_frame = render_mujoco_frame0(episode, width=width, height=height)
    mj_mask = mj_frame.geom_id != -1

    iou = _iou(gltf_mask, mj_mask)

    per_object_iou: dict[int, float] = {}
    for obj in episode.scene.objects:
        if obj.object_id == 0:
            # gltfworld's seg encoding aliases object_id 0 with background
            # (see gltfworld.render.renderer module docstring); per-object
            # IoU isn't meaningful for it from the seg channel alone.
            continue
        geom_ids = [gid for gid, oid in mj_frame.geom_id_to_object_id.items() if oid == obj.object_id]
        if not geom_ids:
            continue
        gltf_obj_mask = gltf_frame.seg == obj.object_id
        mj_obj_mask = np.isin(mj_frame.geom_id, geom_ids)
        per_object_iou[obj.object_id] = _iou(gltf_obj_mask, mj_obj_mask)

    return CrosscheckResult(
        iou=iou,
        per_object_iou=per_object_iou,
        gltf_rgb=gltf_frame.rgb,
        mujoco_rgb=mj_frame.rgb,
        gltf_mask=gltf_mask,
        mujoco_mask=mj_mask,
    )


def write_side_by_side_png(result: CrosscheckResult, out_path: str | Path) -> Path:
    """Write a side-by-side (gltfworld | mujoco | mask diff) PNG for a human to eyeball."""
    from PIL import Image

    h, w = result.gltf_rgb.shape[:2]
    panel = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    panel[:, 0:w] = result.gltf_rgb
    panel[:, w + 10 : 2 * w + 10] = result.mujoco_rgb

    diff = np.zeros((h, w, 3), dtype=np.uint8)
    both = result.gltf_mask & result.mujoco_mask
    only_gltf = result.gltf_mask & ~result.mujoco_mask
    only_mj = result.mujoco_mask & ~result.gltf_mask
    diff[both] = (255, 255, 255)
    diff[only_gltf] = (255, 0, 0)
    diff[only_mj] = (0, 0, 255)
    panel[:, 2 * w + 20 : 3 * w + 20] = diff

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(out_path)
    return out_path
