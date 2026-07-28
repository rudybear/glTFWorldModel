"""Headless glTF-episode rendering (rgb + segmentation + depth), via the
vendored ``pyrender`` (``gltfworld._vendor.pyrender``; see
``src/gltfworld/_vendor/PROVENANCE.md`` for the pinned commit and patch list).

Nothing outside this module (and ``gltfworld._vendor``) should import
``pyrender`` directly -- go through :class:`EpisodeRenderer` /
:func:`render_episode` instead.

EGL setup (module import side effect, must happen *before* the vendored
package is imported):

- ``PYOPENGL_PLATFORM`` is set to ``"egl"`` if not already set by the
  caller's environment, so pyrender always picks the headless EGL platform
  (see ``pyrender.offscreen.OffscreenRenderer._create``) rather than trying
  a windowed pyglet context.
- If ``__EGL_VENDOR_LIBRARY_FILENAMES`` is not already set by the caller,
  this module globs ``/usr/share/glvnd/egl_vendor.d/*.json`` for an ICD
  whose filename mentions "nvidia" and, if found, points
  ``__EGL_VENDOR_LIBRARY_FILENAMES`` at it. Without this, libglvnd's default
  multi-ICD resolution can end up handing the EGL device off to a
  software/Mesa ICD instead of the NVIDIA one depending on install order;
  forcing the NVIDIA ICD (when present) keeps rendering on the GPU. Call
  :func:`egl_info` to see what actually ended up active.

Segmentation encoding: pyrender's ``RenderFlags.SEG`` pass paints each
tracked node with a flat, unlit ``seg_node_map`` color instead of shading it,
so the color IS the label. gltfworld encodes a ``uint16`` ``object_id`` into
that (3,) uint8 color as ``(object_id & 0xFF, (object_id >> 8) & 0xFF, 0)``
(low byte, high byte, blue channel unused) and decodes with
``object_id = r + g * 256``. Background (no geometry hit) reads back as
``(0, 0, 0)`` -> ``object_id == 0``; note this means an ``ObjectSpec`` whose
``object_id`` is legitimately ``0`` (e.g. the ground plane in
``tests/conftest.py:make_sample_episode``) is indistinguishable from
background in the seg buffer -- a pre-existing V1 modeling choice (ground
assigned id 0), not something this encoding can special-case without lying
about a real object's id.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

# --- EGL setup: must run before `gltfworld._vendor.pyrender` is imported ----

_EGL_VENDOR_DIR_GLOB = "/usr/share/glvnd/egl_vendor.d/*.json"


def _set_env_if_unset(key: str, value: str) -> None:
    if key not in os.environ:
        os.environ[key] = value


def _discover_nvidia_egl_vendor_icd() -> str | None:
    """Glob the glvnd vendor directory for an NVIDIA ICD JSON file.

    Never hardcodes a path -- different distros/driver packages name this
    file differently (``10_nvidia.json`` here, but not guaranteed
    elsewhere). Returns ``None`` if no vendor.d directory or no
    nvidia-named file is found (e.g. non-NVIDIA machines), in which case
    EGL vendor selection is left to libglvnd's default behavior.
    """
    for path in sorted(glob.glob(_EGL_VENDOR_DIR_GLOB)):
        if "nvidia" in os.path.basename(path).lower():
            return path
    return None


_set_env_if_unset("PYOPENGL_PLATFORM", "egl")

if os.environ.get("PYOPENGL_PLATFORM") == "egl" and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
    _nvidia_icd = _discover_nvidia_egl_vendor_icd()
    if _nvidia_icd is not None:
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = _nvidia_icd

import numpy as np  # noqa: E402

from gltfworld._vendor import pyrender  # noqa: E402
from gltfworld.scene.episode import Episode  # noqa: E402
from gltfworld.scene.primitives import mesh_for  # noqa: E402

__all__ = ["egl_info", "FrameOutput", "EpisodeRenderer", "render_episode"]


def egl_info() -> dict:
    """Report which EGL/GL vendor+device is actually rendering.

    Creates (and immediately deletes) a tiny 1x1 offscreen context and reads
    back ``GL_VENDOR``/``GL_RENDERER``/``GL_VERSION`` -- this is the only
    way to *prove* which ICD ended up handling the context, since
    environment variables alone only describe what was requested, not what
    was actually granted.
    """
    probe = pyrender.OffscreenRenderer(1, 1)
    try:
        from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

        vendor = glGetString(GL_VENDOR)
        renderer_str = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)
    finally:
        probe.delete()

    return {
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "egl_vendor_library_filenames": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
        "available_vendor_icds": sorted(glob.glob(_EGL_VENDOR_DIR_GLOB)),
        "gl_vendor": vendor.decode() if vendor else None,
        "gl_renderer": renderer_str.decode() if renderer_str else None,
        "gl_version": version.decode() if version else None,
    }


# --- seg encoding ------------------------------------------------------------


def _encode_object_id(object_id: int) -> tuple[int, int, int]:
    """Encode a ``uint16`` object_id as the flat (r, g, b) color pyrender's
    SEG pass paints that node with. See module docstring for the scheme."""
    if not (0 <= object_id <= 0xFFFF):
        raise ValueError(
            f"object_id {object_id} does not fit the uint16 seg encoding (0-65535)"
        )
    return (object_id & 0xFF, (object_id >> 8) & 0xFF, 0)


def _decode_seg(seg_rgb: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_encode_object_id`, vectorized over an (H, W, 3) uint8 image."""
    r = seg_rgb[..., 0].astype(np.uint16)
    g = seg_rgb[..., 1].astype(np.uint16)
    return (r + g * np.uint16(256)).astype(np.uint16)


# --- FrameOutput --------------------------------------------------------------


@dataclass
class FrameOutput:
    """One rendered frame."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32; linear depth in scene units, 0 where no hit
    seg: np.ndarray  # (H, W) uint16; object_id per pixel, 0 = background


# --- EpisodeRenderer ------------------------------------------------------------


class EpisodeRenderer:
    """Renders frames of a gltfworld :class:`~gltfworld.scene.episode.Episode`.

    Builds one pyrender scene per :meth:`load`, reusing a single
    ``pyrender.OffscreenRenderer`` (and GL context) across every
    :meth:`render` call for speed; only object poses are mutated between
    frames (:meth:`set_frame` / :meth:`set_poses`), everything else
    (geometry, materials, camera, lights) is built once.

    Camera aspect ratio deviation: renders are always ``width`` x
    ``height`` square-pixel images, so the camera's projection uses
    ``aspectRatio = width / height`` (not the episode's stored
    ``CameraSpec.aspect``, which describes the *authoring* aspect ratio,
    not this renderer's output). This keeps circular silhouettes circular
    (not stretched) for the analytic tests, which is what a 256x256 render
    calls for regardless of what aspect ratio the camera was authored at.

    Process-lifetime constraint: at most one ``EpisodeRenderer`` should be
    alive-and-in-use per process at a time. pyrender's EGL platform binds
    to the *shared default* EGL display
    (``get_default_device()`` -> ``eglGetDisplay(EGL_DEFAULT_DISPLAY)``),
    and :meth:`delete` (via ``OffscreenRenderer.delete()``) unconditionally
    calls ``eglTerminate()`` on it -- which invalidates that same shared
    display for every *other* still-open ``EpisodeRenderer`` in the
    process (confirmed empirically: a second instance's next ``render()``
    fails with ``EGL_NOT_INITIALIZED`` the moment an unrelated first
    instance is deleted). Not fixed with a patch here since a real fix
    needs global refcounting of the shared display, out of scope for this
    project's minimal pyrender patch set (see
    ``src/gltfworld/_vendor/PROVENANCE.md``); instead, treat one
    ``EpisodeRenderer`` as a process-wide singleton and call
    :meth:`load` again to switch episodes/scenes rather than constructing a
    second instance (this is also just what "reuse ONE OffscreenRenderer
    across frames" already asks for). The test suite shares exactly one
    instance across all gpu-marked tests via the session-scoped
    ``episode_renderer`` fixture in ``tests/conftest.py``.
    """

    def __init__(self, width: int = 256, height: int = 256) -> None:
        self.width = int(width)
        self.height = int(height)
        self._renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self._scene: pyrender.Scene | None = None
        self._episode: Episode | None = None
        self._object_nodes: list[pyrender.Node] = []
        self._seg_node_map: dict[pyrender.Node, tuple[int, int, int]] = {}

    def load(self, episode: Episode) -> None:
        """Build the pyrender scene for ``episode`` and jump to frame 0."""
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[0.0, 0.0, 0.0])

        geometry_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        object_nodes: list[pyrender.Node] = []
        seg_node_map: dict[pyrender.Node, tuple[int, int, int]] = {}

        for i, obj in enumerate(episode.scene.objects):
            geometry_key = (obj.shape, obj.size.tobytes())
            if geometry_key not in geometry_cache:
                positions, normals, indices = mesh_for(obj.shape, obj.size)
                geometry_cache[geometry_key] = (positions, normals, indices.reshape(-1, 3))
            positions, normals, tri_indices = geometry_cache[geometry_key]

            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[float(v) for v in obj.color],
                roughnessFactor=float(obj.roughness),
                metallicFactor=float(obj.metallic),
            )
            primitive = pyrender.Primitive(
                positions=positions, normals=normals, indices=tri_indices, material=material
            )
            mesh = pyrender.Mesh(primitives=[primitive])

            pose0 = episode.series.poses[0, i]
            node = pyrender.Node(
                name=f"obj_{obj.object_id}",
                mesh=mesh,
                translation=np.array(pose0[0:3], dtype=np.float64),
                rotation=np.array(pose0[3:7], dtype=np.float64),
            )
            scene.add_node(node)
            object_nodes.append(node)
            seg_node_map[node] = _encode_object_id(obj.object_id)

        cam = episode.scene.camera
        camera = pyrender.PerspectiveCamera(
            yfov=float(cam.yfov),
            znear=float(cam.znear),
            zfar=float(cam.zfar),
            aspectRatio=self.width / self.height,
        )
        camera_node = pyrender.Node(
            camera=camera,
            translation=np.array(cam.position, dtype=np.float64),
            rotation=np.array(cam.rotation, dtype=np.float64),
        )
        scene.add_node(camera_node)
        scene.main_camera_node = camera_node

        for light in episode.scene.lights:
            if light.type == "directional":
                gl_light = pyrender.DirectionalLight(
                    color=np.array(light.color, dtype=np.float64), intensity=float(light.intensity)
                )
                light_node = pyrender.Node(
                    light=gl_light, rotation=np.array(light.rotation, dtype=np.float64)
                )
            else:
                gl_light = pyrender.PointLight(
                    color=np.array(light.color, dtype=np.float64), intensity=float(light.intensity)
                )
                light_node = pyrender.Node(
                    light=gl_light, translation=np.array(light.position, dtype=np.float64)
                )
            scene.add_node(light_node)

        self._scene = scene
        self._episode = episode
        self._object_nodes = object_nodes
        self._seg_node_map = seg_node_map

    def _require_loaded(self) -> tuple[pyrender.Scene, Episode]:
        if self._scene is None or self._episode is None:
            raise RuntimeError("EpisodeRenderer.load(episode) must be called first")
        return self._scene, self._episode

    def _apply_poses(self, poses: np.ndarray) -> None:
        for node, pose in zip(self._object_nodes, poses):
            node.translation = np.array(pose[0:3], dtype=np.float64)
            node.rotation = np.array(pose[3:7], dtype=np.float64)

    def set_frame(self, t: int) -> None:
        """Move every object to its pose at frame ``t`` of the loaded episode."""
        _, episode = self._require_loaded()
        self._apply_poses(episode.series.poses[t])

    def set_poses(self, poses: np.ndarray) -> None:
        """Move every object to an arbitrary (N, 7) xyz+quat(xyzw) pose array
        (N must match the number of objects passed to :meth:`load`)."""
        self._require_loaded()
        poses = np.asarray(poses, dtype=np.float32)
        if poses.shape != (len(self._object_nodes), 7):
            raise ValueError(
                f"expected poses shape ({len(self._object_nodes)}, 7), got {poses.shape}"
            )
        self._apply_poses(poses)

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass: (rgb (H,W,3) uint8, depth (H,W) float32)."""
        scene, _ = self._require_loaded()
        color, depth = self._renderer.render(scene)
        return color, depth

    def _render_depth_only(self) -> np.ndarray:
        """Depth-only forward pass: skips the lighting bind and RGB
        ``glReadPixels`` call that a full ``render_rgbd()`` does. Not part
        of the public rgb/seg/depth API -- exists so benchmarks can isolate
        "draw+shade" cost from "color readback" cost (see
        ``tests/test_render_bench.py``)."""
        scene, _ = self._require_loaded()
        return self._renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

    def render_seg(self) -> np.ndarray:
        """One SEG forward pass, decoded to (H,W) uint16 object ids."""
        scene, _ = self._require_loaded()
        seg_color, _ = self._renderer.render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map=self._seg_node_map
        )
        return _decode_seg(seg_color)

    def render(self) -> FrameOutput:
        """Render the current frame: rgb + depth (one pass) + seg (one pass)."""
        color, depth = self.render_rgbd()
        seg = self.render_seg()
        return FrameOutput(
            rgb=np.ascontiguousarray(color, dtype=np.uint8),
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            seg=np.ascontiguousarray(seg, dtype=np.uint16),
        )

    def delete(self) -> None:
        """Free the underlying OpenGL context/renderer."""
        self._renderer.delete()

    def __enter__(self) -> "EpisodeRenderer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.delete()


def render_episode(path_glb, out_dir, width: int = 256, height: int = 256):
    """Render every frame of the episode GLB at ``path_glb`` and write:

    - ``rgb.npy``   (T, H, W, 3) uint8
    - ``seg.npy``   (T, H, W) uint16
    - ``depth.npy`` (T, H, W) float16
    - ``frame_000.png`` -- a preview of frame 0's rgb

    into ``out_dir`` (created if missing). Returns ``out_dir`` as a ``Path``.
    """
    from pathlib import Path

    from PIL import Image

    from gltfworld.scene.convert import load_episode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = load_episode(path_glb)
    num_frames = episode.series.num_frames

    renderer = EpisodeRenderer(width=width, height=height)
    try:
        renderer.load(episode)

        rgb = np.empty((num_frames, height, width, 3), dtype=np.uint8)
        seg = np.empty((num_frames, height, width), dtype=np.uint16)
        depth = np.empty((num_frames, height, width), dtype=np.float16)

        for t in range(num_frames):
            renderer.set_frame(t)
            frame = renderer.render()
            rgb[t] = frame.rgb
            seg[t] = frame.seg
            depth[t] = frame.depth.astype(np.float16)

        np.save(out_dir / "rgb.npy", rgb)
        np.save(out_dir / "seg.npy", seg)
        np.save(out_dir / "depth.npy", depth)
        Image.fromarray(rgb[0]).save(out_dir / "frame_000.png")
    finally:
        renderer.delete()

    return out_dir
