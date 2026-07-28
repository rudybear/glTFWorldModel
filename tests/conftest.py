"""Shared test fixtures: deterministic sample episode + JSON Schema helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.scene import CameraSpec, LightSpec, ObjectSpec, SceneState

REPO_ROOT = Path(__file__).resolve().parent.parent
KHR_SCHEMA_DIR = REPO_ROOT / "docs" / "schemas" / "khr"
RWM_SCHEMA_DIR = REPO_ROOT / "docs" / "schemas" / "rwm"

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_SHAPES_CYCLE = ("sphere", "box", "cylinder")


def make_sample_episode(n_objects: int = 3, T: int = 30) -> Episode:
    """A deterministic sample episode: falling sphere/box/cylinder over a static ground box.

    Kinematics are hand-rolled free-fall under ``scene.gravity`` (no collision
    response) -- enough to exercise the transport codec end-to-end; real
    physics (MuJoCo) lands in milestone V3.
    """
    dt = 1.0 / 60.0
    gravity_y = -9.81

    ground = ObjectSpec(
        object_id=0,
        shape="box",
        size=np.array([5.0, 0.1, 5.0], dtype=np.float32),
        color=np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1.0,
        friction=0.6,
        restitution=0.1,
        is_static=True,
        category="ground",
    )
    objects = [ground]
    start_heights = []

    for i in range(n_objects):
        shape = _SHAPES_CYCLE[i % len(_SHAPES_CYCLE)]
        object_id = i + 1
        if shape == "sphere":
            size = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            category = "ball"
        elif shape == "box":
            size = np.array([0.25, 0.25, 0.25], dtype=np.float32)
            category = "crate"
        else:
            size = np.array([0.2, 0.3, 0.2], dtype=np.float32)
            category = "crate"
        objects.append(
            ObjectSpec(
                object_id=object_id,
                shape=shape,
                size=size,
                color=np.array([0.1 + 0.1 * i, 0.4, 0.8, 1.0], dtype=np.float32),
                roughness=0.4,
                metallic=0.0,
                mass=1.0 + 0.5 * i,
                friction=0.5,
                restitution=0.3,
                is_static=False,
                category=category,
            )
        )
        start_heights.append(2.0 + i * 0.5)

    camera = CameraSpec(
        position=np.array([0.0, 2.0, 6.0], dtype=np.float32),
        rotation=_IDENTITY_QUAT.copy(),
        yfov=0.8,
        znear=0.05,
        zfar=100.0,
        aspect=16.0 / 9.0,
    )
    lights = [
        LightSpec(
            type="directional",
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=3.0,
            rotation=_IDENTITY_QUAT.copy(),
        ),
        LightSpec(
            type="point",
            color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
            intensity=10.0,
            position=np.array([2.0, 3.0, 1.0], dtype=np.float32),
        ),
    ]

    scene = SceneState(
        objects=objects,
        camera=camera,
        lights=lights,
        gravity=np.array([0.0, gravity_y, 0.0], dtype=np.float32),
        dt=dt,
        seed=1234,
    )

    n = len(objects)
    times = (np.arange(T, dtype=np.float32) * np.float32(dt)).astype(np.float32)
    poses = np.zeros((T, n, 7), dtype=np.float32)
    lin_vel = np.zeros((T, n, 3), dtype=np.float32)
    ang_vel = np.zeros((T, n, 3), dtype=np.float32)

    poses[:, 0, 3:7] = _IDENTITY_QUAT  # ground stays put at the origin

    for i in range(1, n):
        h0 = start_heights[i - 1]
        x = float(i)
        for t in range(T):
            time = times[t]
            vy = gravity_y * time
            y = h0 + 0.5 * gravity_y * time * time
            poses[t, i, 0] = x
            poses[t, i, 1] = y
            poses[t, i, 2] = 0.0
            poses[t, i, 3:7] = _IDENTITY_QUAT
            lin_vel[t, i, 1] = vy

    series = StateSeries(times=times, poses=poses, lin_vel=lin_vel, ang_vel=ang_vel)
    return Episode(scene=scene, series=series)


@pytest.fixture
def sample_episode() -> Episode:
    return make_sample_episode()


@pytest.fixture(scope="session")
def episode_renderer():
    """One process-lifetime ``EpisodeRenderer``, shared by every gpu-marked
    render test in the session.

    This isn't just an optimization: pyrender's EGL platform binds to the
    *shared default* EGL display (``get_default_device()`` ->
    ``eglGetDisplay(EGL_DEFAULT_DISPLAY)``), and
    ``OffscreenRenderer.delete()`` unconditionally calls ``eglTerminate()``
    on it. Deleting *any* one ``EpisodeRenderer``/``OffscreenRenderer``
    instance in a process invalidates that shared display for every other
    still-open instance (confirmed empirically: a second renderer's next
    ``render()`` call fails with ``EGL_NOT_INITIALIZED`` as soon as a first,
    unrelated renderer is deleted) -- not something a small, targeted patch
    can fix without global refcounting of the shared display, which is out
    of scope for gltfworld's pyrender patch set (see
    ``src/gltfworld/_vendor/PROVENANCE.md``). So: exactly one
    ``EpisodeRenderer`` may be alive-and-in-use per process; every test that
    needs one shares this fixture and calls ``.load(episode)`` fresh
    (cheap -- it only rebuilds the scene graph, not the GL context).
    """
    pytest.importorskip("gltfworld.render.renderer")
    from gltfworld.render.renderer import EpisodeRenderer

    renderer = EpisodeRenderer(width=256, height=256)
    yield renderer
    renderer.delete()


def _load_schema_registry(root: Path) -> Registry:
    resources = []
    for path in sorted(root.rglob("*.schema.json")):
        with open(path) as f:
            schema = json.load(f)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@pytest.fixture(scope="session")
def khr_schema_registry() -> Registry:
    return _load_schema_registry(KHR_SCHEMA_DIR)


@pytest.fixture(scope="session")
def rwm_schema_registry() -> Registry:
    return _load_schema_registry(RWM_SCHEMA_DIR)


def validate_against_schema(schema_path: Path, registry: Registry | None, instance) -> None:
    """Validate ``instance`` against the JSON Schema at ``schema_path``, resolving
    ``$ref``s through ``registry`` (built from all vendored schemas' ``$id``s).

    ``registry=None`` is fine for self-contained schemas (no external $refs).
    """
    with open(schema_path) as f:
        schema = json.load(f)
    Draft202012Validator(schema, registry=registry or Registry()).validate(instance)
