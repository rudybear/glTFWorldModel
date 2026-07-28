# Vendored: pyrender

- **Repo**: https://github.com/mmatl/pyrender
- **Pinned commit**: `a59963ef890891656fd17c90e12d663233dcaa99` (latest `master` as of
  vendoring; authored 2022-04-30)
- **Vendored on**: 2026-07-27
- **Vendored path**: `src/gltfworld/_vendor/pyrender/` (the `pyrender/` package
  directory from the repo root, plus its `LICENSE`; repo-root `tests/`,
  `docs/`, `examples/`, `setup.py` etc. are not vendored)
- **License**: MIT (see `pyrender/LICENSE` alongside this file)

Nothing outside `gltfworld._vendor` may import `pyrender` directly; all
access goes through `gltfworld.render.renderer`, which is responsible for
setting `PYOPENGL_PLATFORM=egl` (and forcing the NVIDIA EGL ICD if needed)
*before* the vendored package is imported.

## Why vendored, not a normal dependency

Pinned in-tree so it can be patched (see below) for numpy 2 / Python 3.12
compatibility and to make the `pyglet` (interactive viewer) dependency truly
optional for headless/offscreen use, without waiting on upstream (last
released 2022, appears unmaintained for these issues).

## Patch set (full list)

All patches are minimal, targeted diffs against the pinned commit above.

1. **`pyrender/__init__.py`** — guard the `Viewer` import.
   - **What**: wrapped `from .viewer import Viewer` in
     `try/except ImportError: Viewer = None`.
   - **Why**: `pyrender/__init__.py` unconditionally imports `.viewer`,
     which does `import pyglet` at module scope. `pyglet` is only needed
     for the interactive on-screen `Viewer`; it is not installed (and not
     listed in gltfworld's `render` extra, see pyproject.toml) since
     gltfworld only ever uses `OffscreenRenderer` for headless rendering.
     Without this guard, `import pyrender` itself fails with
     `ModuleNotFoundError: No module named 'pyglet'` before any offscreen
     rendering code can even run. `pyrender.offscreen.OffscreenRenderer`
     already lazily imports `pyrender.platforms.pyglet_platform` (only
     when `PYOPENGL_PLATFORM` is unset) and `pyrender.platforms.egl` (only
     when `PYOPENGL_PLATFORM == "egl"`), so neither of those modules is
     actually reached at runtime once `gltfworld.render.renderer` forces
     `PYOPENGL_PLATFORM=egl` before import; only the package `__init__.py`
     import chain needed guarding.

2. **`pyrender/mesh.py`** — replace `np.infty` with `np.inf`.
   - **What**: in `Mesh.bounds` (the two-line initial min/max accumulator),
     replaced both `np.infty` occurrences with `np.inf`.
   - **Why**: `np.infty` was removed in numpy 2.0 (this project pins
     `numpy>=1.26`, and the installed/tested version is numpy 2.x);
     accessing it raises `AttributeError: module 'numpy' has no attribute
     'infty'`. `Mesh.bounds` is reached via `Scene.bounds` /
     `Scene.scale` / `Scene.centroid`, which the renderer's shadow-mapping
     pass (`Renderer._shadow_mapping_pass`, via `scene.scale`) calls
     whenever a `SHADOWS_*` render flag is set and a light in the scene
     requests a shadow pass — a real, reachable path (gltfworld scenes
     always include at least one directional light), not just a
     theoretical one.

3. **`pyrender/offscreen.py`** and **`pyrender/platforms/pyglet_platform.py`**
   — fix intra-package imports broken by vendoring.
   - **What**: `offscreen.py`'s `OffscreenRenderer._create()` had three
     absolute imports (`from pyrender.platforms.pyglet_platform import
     PygletPlatform`, `from pyrender.platforms import egl`, `from
     pyrender.platforms.osmesa import OSMesaPlatform`), changed to relative
     (`from .platforms.pyglet_platform import ...`, `from .platforms import
     egl`, `from .platforms.osmesa import ...`).
     `platforms/pyglet_platform.py` had `from pyrender.constants import
     (...)`, changed to `from ..constants import (...)`.
   - **Why**: upstream assumes `pyrender` is installed as a top-level
     package (`import pyrender` resolves to a real top-level module), so it
     mixes relative (`from .base import Platform`) and absolute
     (`from pyrender.constants import ...`) intra-package imports
     interchangeably. Vendored under `gltfworld._vendor.pyrender`, there is
     no top-level `pyrender` module, so the absolute forms raised
     `ModuleNotFoundError: No module named 'pyrender'` the moment
     `OffscreenRenderer._create()` (or, for `pyglet_platform.py`, its
     lazy-imported module body) actually ran — i.e. every real render call
     with `PYOPENGL_PLATFORM=egl` set, not merely `import pyrender` itself.

No other numpy-2/Python-3.12 breakage was found: a targeted grep across the
vendored tree for other removed numpy aliases (`np.float`, `np.int`,
`np.bool`, `np.object`, `np.complex`, `np.str`, `np.long`) and for
Python-version-sensitive constructs (`collections.Mapping`-style ABC
imports, `distutils`/`pkg_resources` usage) found nothing else. The
patched package was exercised end-to-end on this machine (EGL/NVIDIA,
Python 3.12.3, numpy 2.x) covering: plain color+depth render, directional
light with `SHADOWS_DIRECTIONAL` (exercises patch 2), and `RenderFlags.SEG`
segmentation render (exercises patch 1's import path indirectly, since
`gltfworld.render.renderer` imports the package once at module load).

## `render` extra dependency set

`pyrender`'s own `setup.py` lists: `freetype-py`, `imageio`, `networkx`,
`numpy`, `Pillow`, `pyglet>=1.4.10`, `PyOpenGL~=3.1.0`, `scipy`, `six`,
`trimesh`. gltfworld's `render` extra (see `pyproject.toml`) only lists
what the vendored code actually imports on the code paths gltfworld
exercises:

- `PyOpenGL` — `OpenGL.GL`/`OpenGL.EGL`/`OpenGL.platform` throughout
  (`renderer.py`, `primitive.py`, `texture.py`, `shader_program.py`,
  `platforms/egl.py`, ...).
- `Pillow` — `utils.py` (`PIL.Image`, texture format conversion) and
  `renderer.py` (`import PIL`, macOS resize path).
- `freetype-py` — `font.py` (`import freetype`), imported unconditionally
  by `renderer.py` (`FontCache`) even though gltfworld never calls
  `render_text`.
- `networkx` — `scene.py` (`import networkx as nx`, the scene graph).
- `six` — `camera.py`/`light.py`/`material.py`/`platforms/base.py`
  (`six.add_metaclass` for Python 2/3 ABC compatibility shims).

Deliberately **excluded**, despite being in upstream's `install_requires`:

- **`pyglet`** — only needed by the interactive `Viewer`, guarded out by
  patch 1 above; gltfworld only uses `OffscreenRenderer`.
- **`imageio`** — only imported by `viewer.py` (unreachable once `Viewer`
  import is guarded/skipped); no other vendored module imports it.
- **`scipy`** — not imported directly by any vendored `pyrender` module
  (grepped the whole tree). Upstream's `setup.py` includes it only with the
  comment "because of trimesh missing dep" — i.e. it backs an *optional*
  code path inside `trimesh` (weighted-sparse `Trimesh.vertex_normals`),
  which gltfworld's renderer never triggers: `gltfworld.render.renderer`
  builds `pyrender.Primitive`s directly from
  `gltfworld.scene.primitives.mesh_for`'s own (positions, normals, indices)
  arrays rather than going through `pyrender.Mesh.from_trimesh`, the only
  vendored entry point that would touch trimesh mesh objects. `trimesh`
  itself is already a core gltfworld dependency (`pyproject.toml`
  `dependencies`), independent of this extra.
