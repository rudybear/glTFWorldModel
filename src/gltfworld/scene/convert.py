"""THE glTF transport codec: ``Episode <-> pygltflib.GLTF2`` (GLB/.gltf).

Encoding layout (see DESIGN.md "Transport encoding" for the prose writeup):

- One node per object (name ``obj_{object_id}``), TRS = frame-0 pose.
  Mesh geometry is deduplicated by ``(shape, size bytes)``; materials are
  built from ``ObjectSpec``'s pbr fields.
- One camera node + camera; lights via ``KHR_lights_punctual``.
- ``animations[0]``: one shared time accessor, two STEP channels
  (translation, rotation) per object. This is the sole place poses live on
  the wire -- node TRS only mirrors frame 0.
- ``KHR_physics_rigid_bodies`` + ``KHR_implicit_shapes`` carry per-object
  physics (mass/friction/restitution/collider shape), with initial
  velocities taken from ``series.lin_vel``/``series.ang_vel`` at t=0.
- ``RWM_state_series`` (custom vendor extension) carries the full
  velocity/action/pose-variance time series, reusing the same time
  accessor. ``extras.rwm`` carries object/scene bookkeeping glTF has no
  standard home for.
- ``extensionsUsed`` lists every extension actually written;
  ``extensionsRequired`` is always left empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib

from gltfworld.ext import khr_physics, rwm
from gltfworld.gltf.accessors import BufferAccumulator, read_accessor
from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.primitives import mesh_for
from gltfworld.scene.scene import ArticulatedSpec, CameraSpec, LightSpec, ObjectSpec, SceneState

EXT_LIGHTS_PUNCTUAL = "KHR_lights_punctual"

GENERATOR = "gltfworld"


# --- V9-prep: articulated joint pivots -----------------------------------------
#
# See gltfworld.ext.khr_physics's module docstring ("V9-prep: joints") for
# why each articulation gets one motion-less, geometry-less "joint pivot"
# child node per side (base/part), both placed at ArticulatedSpec.anchor in
# their own body's local frame: that's what makes the pinned spec's
# attachment-frame rule ("relative transform between the node and the first
# parent motion, or the fixed reference frame if none") land exactly on the
# shared physical hinge/slide point, rather than approximating it.


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` (3,) by unit quaternion ``q`` (4,) xyzw."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = q[0:3]
    w = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _world_to_local_offset(world_point: np.ndarray, body_pos: np.ndarray, body_rot_xyzw: np.ndarray) -> np.ndarray:
    """``world_point`` expressed in ``body``'s own local frame (its inverse
    rotation applied to the world-space offset from its origin)."""
    offset_world = np.asarray(world_point, dtype=np.float64) - np.asarray(body_pos, dtype=np.float64)
    return _quat_rotate_xyzw(_quat_conj_xyzw(body_rot_xyzw), offset_world)


def _add_articulation_pivots(
    gltf: pygltflib.GLTF2,
    scene: SceneState,
    series,
    node_index_by_object_id: dict[int, int],
) -> list[tuple[ArticulatedSpec, int, int]]:
    """Add one motion-less "joint pivot" child node per side (base/part) for
    every ``scene.articulations`` entry; nests each under its owning
    object's node (``node.children``). Returns
    ``[(articulation, base_pivot_node_index, part_pivot_node_index), ...]``
    for the caller to turn into KHR joint dicts once the physics doc exists.
    """
    obj_position_by_id = {obj.object_id: i for i, obj in enumerate(scene.objects)}
    out: list[tuple[ArticulatedSpec, int, int]] = []

    for art in scene.articulations:
        base_node_index = node_index_by_object_id[art.base_object_id]
        part_node_index = node_index_by_object_id[art.part_object_id]
        base_pos0 = series.poses[0, obj_position_by_id[art.base_object_id], 0:3]
        base_rot0 = series.poses[0, obj_position_by_id[art.base_object_id], 3:7]
        part_pos0 = series.poses[0, obj_position_by_id[art.part_object_id], 0:3]
        part_rot0 = series.poses[0, obj_position_by_id[art.part_object_id], 3:7]

        base_local = _world_to_local_offset(art.anchor, base_pos0, base_rot0)
        part_local = _world_to_local_offset(art.anchor, part_pos0, part_rot0)

        base_pivot = pygltflib.Node(
            name=f"joint_anchor_base_{art.joint_index}",
            translation=[float(v) for v in base_local],
            rotation=[0.0, 0.0, 0.0, 1.0],
        )
        gltf.nodes.append(base_pivot)
        base_pivot_index = len(gltf.nodes) - 1

        part_pivot = pygltflib.Node(
            name=f"joint_anchor_part_{art.joint_index}",
            translation=[float(v) for v in part_local],
            rotation=[0.0, 0.0, 0.0, 1.0],
        )
        gltf.nodes.append(part_pivot)
        part_pivot_index = len(gltf.nodes) - 1

        base_node = gltf.nodes[base_node_index]
        base_node.children = list(base_node.children or []) + [base_pivot_index]
        part_node = gltf.nodes[part_node_index]
        part_node.children = list(part_node.children or []) + [part_pivot_index]

        out.append((art, base_pivot_index, part_pivot_index))

    return out


def _compute_scene_roots(gltf: pygltflib.GLTF2) -> list[int]:
    """Every node index NOT listed as a child of some other node -- the
    correct ``scenes[0].nodes`` root list once any node (e.g. an
    articulation joint pivot) is nested under a parent via
    ``node.children``. Identical to ``list(range(len(gltf.nodes)))`` when
    nothing has children (every pre-V9 episode)."""
    child_indices: set[int] = set()
    for node in gltf.nodes:
        if node.children:
            child_indices.update(node.children)
    return [i for i in range(len(gltf.nodes)) if i not in child_indices]


def articulation_to_extras(art: ArticulatedSpec) -> dict:
    """Raw-dict encoding of one ``ArticulatedSpec``, for ``extras.rwm.articulations``."""
    return {
        "joint_index": art.joint_index,
        "base_object_id": art.base_object_id,
        "part_object_id": art.part_object_id,
        "joint_type": art.joint_type,
        "axis": art.axis,
        "min": art.min,
        "max": art.max,
        "anchor": [float(v) for v in art.anchor],
        "part_labels": list(art.part_labels),
        "affordances": list(art.affordances),
        "handle_object_id": art.handle_object_id,
        "handle_labels": list(art.handle_labels),
        "handle_affordances": list(art.handle_affordances),
        "base_labels": list(art.base_labels),
    }


def articulation_from_extras(d: dict) -> ArticulatedSpec:
    """Inverse of :func:`articulation_to_extras`."""
    return ArticulatedSpec(
        joint_index=d["joint_index"],
        base_object_id=d["base_object_id"],
        part_object_id=d["part_object_id"],
        joint_type=d["joint_type"],
        axis=d["axis"],
        min=d["min"],
        max=d["max"],
        anchor=np.array(d["anchor"], dtype=np.float32),
        part_labels=tuple(d.get("part_labels", ())),
        affordances=tuple(d.get("affordances", ())),
        handle_object_id=d.get("handle_object_id"),
        handle_labels=tuple(d.get("handle_labels", ())),
        handle_affordances=tuple(d.get("handle_affordances", ())),
        base_labels=tuple(d.get("base_labels", ())),
    )


def _semantics_by_object_id(scene: SceneState) -> dict[int, dict]:
    """object_id -> ``extras.rwm.semantics`` dict for every base/part/handle
    object across ``scene.articulations`` (empty for non-articulated scenes)."""
    out: dict[int, dict] = {}
    for art in scene.articulations:
        out[art.base_object_id] = rwm.node_semantics(art.base_labels or ("base",))
        out[art.part_object_id] = rwm.node_semantics(art.part_labels, art.affordances)
        if art.handle_object_id is not None:
            out[art.handle_object_id] = rwm.node_semantics(
                art.handle_labels or ("handle",), art.handle_affordances
            )
    return out


# --- encode -------------------------------------------------------------------


def _geometry_key(obj: ObjectSpec) -> tuple:
    return (obj.shape, obj.size.tobytes())


def _material_key(obj: ObjectSpec) -> tuple:
    return (obj.color.tobytes(), obj.roughness, obj.metallic)


def _add_geometry(
    gltf: pygltflib.GLTF2,
    accumulator: BufferAccumulator,
    cache: dict[tuple, tuple[int, int, int]],
    obj: ObjectSpec,
) -> tuple[int, int, int]:
    key = _geometry_key(obj)
    if key in cache:
        return cache[key]

    positions, normals, indices = mesh_for(obj.shape, obj.size)
    pos_acc = accumulator.add_accessor(
        gltf, positions, type_=pygltflib.VEC3, target=pygltflib.ARRAY_BUFFER, compute_minmax=True
    )
    normal_acc = accumulator.add_accessor(
        gltf, normals, type_=pygltflib.VEC3, target=pygltflib.ARRAY_BUFFER
    )
    indices_acc = accumulator.add_accessor(
        gltf, indices, type_=pygltflib.SCALAR, target=pygltflib.ELEMENT_ARRAY_BUFFER
    )
    cache[key] = (pos_acc, normal_acc, indices_acc)
    return cache[key]


def _add_material(gltf: pygltflib.GLTF2, cache: dict[tuple, int], obj: ObjectSpec) -> int:
    key = _material_key(obj)
    if key in cache:
        return cache[key]

    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[float(v) for v in obj.color],
            roughnessFactor=obj.roughness,
            metallicFactor=obj.metallic,
        ),
        name=f"mat_{obj.category}",
    )
    gltf.materials.append(material)
    index = len(gltf.materials) - 1
    cache[key] = index
    return index


def _add_object_node(
    gltf: pygltflib.GLTF2,
    accumulator: BufferAccumulator,
    geometry_cache: dict[tuple, tuple[int, int, int]],
    material_cache: dict[tuple, int],
    obj: ObjectSpec,
    pose0: np.ndarray,
) -> int:
    pos_acc, normal_acc, indices_acc = _add_geometry(gltf, accumulator, geometry_cache, obj)
    material_index = _add_material(gltf, material_cache, obj)

    mesh = pygltflib.Mesh(
        primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=pos_acc, NORMAL=normal_acc),
                indices=indices_acc,
                material=material_index,
                mode=pygltflib.TRIANGLES,
            )
        ]
    )
    gltf.meshes.append(mesh)
    mesh_index = len(gltf.meshes) - 1

    node = pygltflib.Node(
        name=f"obj_{obj.object_id}",
        mesh=mesh_index,
        translation=[float(v) for v in pose0[0:3]],
        rotation=[float(v) for v in pose0[3:7]],
    )
    gltf.nodes.append(node)
    return len(gltf.nodes) - 1


def _add_camera_node(gltf: pygltflib.GLTF2, camera: CameraSpec) -> int:
    gltf.cameras.append(
        pygltflib.Camera(
            type="perspective",
            perspective=pygltflib.Perspective(
                aspectRatio=camera.aspect,
                yfov=camera.yfov,
                znear=camera.znear,
                zfar=camera.zfar,
            ),
        )
    )
    camera_index = len(gltf.cameras) - 1
    node = pygltflib.Node(
        name="camera",
        camera=camera_index,
        translation=[float(v) for v in camera.position],
        rotation=[float(v) for v in camera.rotation],
    )
    gltf.nodes.append(node)
    return len(gltf.nodes) - 1


def _add_light_nodes(gltf: pygltflib.GLTF2, lights: list[LightSpec]) -> list[int]:
    if not lights:
        return []

    light_dicts = []
    node_indices = []
    for i, light in enumerate(lights):
        light_dicts.append(
            {
                "type": light.type,
                "color": [float(v) for v in light.color],
                "intensity": light.intensity,
            }
        )
        node = pygltflib.Node(name=f"light_{i}")
        if light.type == "directional":
            node.rotation = [float(v) for v in light.rotation]
        else:
            node.translation = [float(v) for v in light.position]
        node.extensions[EXT_LIGHTS_PUNCTUAL] = {"light": i}
        gltf.nodes.append(node)
        node_indices.append(len(gltf.nodes) - 1)

    gltf.extensions.setdefault(EXT_LIGHTS_PUNCTUAL, {})
    gltf.extensions[EXT_LIGHTS_PUNCTUAL]["lights"] = light_dicts
    return node_indices


def episode_to_gltf(ep: Episode) -> pygltflib.GLTF2:
    """Encode ``ep`` into a fresh ``pygltflib.GLTF2`` document."""
    scene = ep.scene
    series = ep.series

    gltf = pygltflib.GLTF2()
    gltf.asset = pygltflib.Asset(generator=GENERATOR, version="2.0")
    accumulator = BufferAccumulator()

    geometry_cache: dict[tuple, tuple[int, int, int]] = {}
    material_cache: dict[tuple, int] = {}
    node_index_by_object_id: dict[int, int] = {}
    object_ids_in_order = [obj.object_id for obj in scene.objects]

    for i, obj in enumerate(scene.objects):
        pose0 = series.poses[0, i]
        node_index = _add_object_node(gltf, accumulator, geometry_cache, material_cache, obj, pose0)
        node_index_by_object_id[obj.object_id] = node_index

    _add_camera_node(gltf, scene.camera)
    _add_light_nodes(gltf, scene.lights)

    # --- V9-prep: articulation joint pivots (see module's top-of-file note) ---
    articulation_pivots = _add_articulation_pivots(gltf, scene, series, node_index_by_object_id)

    gltf.scenes = [pygltflib.Scene(nodes=_compute_scene_roots(gltf))]
    gltf.scene = 0

    # --- animation: one shared time accessor, STEP translation+rotation per object ---
    # compute_minmax=True: required by the glTF spec for animation sampler
    # input accessors (and by the validator: ANIMATION_SAMPLER_INPUT_ACCESSOR_WITHOUT_BOUNDS).
    times_accessor = accumulator.add_accessor(gltf, series.times, type_=pygltflib.SCALAR, compute_minmax=True)

    animation = pygltflib.Animation()
    for i, obj in enumerate(scene.objects):
        node_index = node_index_by_object_id[obj.object_id]
        translations = np.ascontiguousarray(series.poses[:, i, 0:3])
        rotations = np.ascontiguousarray(series.poses[:, i, 3:7])

        trans_acc = accumulator.add_accessor(gltf, translations, type_=pygltflib.VEC3)
        rot_acc = accumulator.add_accessor(gltf, rotations, type_=pygltflib.VEC4)

        animation.samplers.append(
            pygltflib.AnimationSampler(input=times_accessor, output=trans_acc, interpolation="STEP")
        )
        trans_sampler_index = len(animation.samplers) - 1
        animation.samplers.append(
            pygltflib.AnimationSampler(input=times_accessor, output=rot_acc, interpolation="STEP")
        )
        rot_sampler_index = len(animation.samplers) - 1

        animation.channels.append(
            pygltflib.AnimationChannel(
                sampler=trans_sampler_index,
                target=pygltflib.AnimationChannelTarget(node=node_index, path="translation"),
            )
        )
        animation.channels.append(
            pygltflib.AnimationChannel(
                sampler=rot_sampler_index,
                target=pygltflib.AnimationChannelTarget(node=node_index, path="rotation"),
            )
        )
    gltf.animations = [animation]

    # --- KHR_physics_rigid_bodies + KHR_implicit_shapes ---
    initial_lin_vel: dict[int, np.ndarray] = {}
    initial_ang_vel: dict[int, np.ndarray] = {}
    for i, obj in enumerate(scene.objects):
        if series.lin_vel is not None:
            initial_lin_vel[obj.object_id] = series.lin_vel[0, i]
        if series.ang_vel is not None:
            initial_ang_vel[obj.object_id] = series.ang_vel[0, i]

    physics_doc = khr_physics.build_khr_physics(
        scene.objects, node_index_by_object_id, initial_lin_vel, initial_ang_vel
    )

    # --- V9-prep: KHR_physics_rigid_bodies.physicsJoints[] + node.joint ---
    for art, base_pivot_index, part_pivot_index in articulation_pivots:
        if art.joint_type == "revolute":
            limits = khr_physics.hinge_joint_limits(art.axis, art.min, art.max)
        else:
            limits = khr_physics.slider_joint_limits(art.axis, art.min, art.max)
        physics_doc.joints.append(khr_physics.build_joint_dict(limits))
        joint_array_index = len(physics_doc.joints) - 1
        physics_doc.node_physics[part_pivot_index] = {
            "joint": khr_physics.node_joint_property(base_pivot_index, joint_array_index)
        }

    khr_physics.write_khr_physics(gltf, physics_doc)

    # --- RWM_state_series (reuses times_accessor) ---
    rwm_doc = rwm.build_state_series(
        gltf, accumulator, times_accessor, node_index_by_object_id, object_ids_in_order, series
    )
    rwm.write_state_series(gltf, rwm_doc)

    # --- extras.rwm ---
    semantics_by_object_id = _semantics_by_object_id(scene)
    for obj in scene.objects:
        node = gltf.nodes[node_index_by_object_id[obj.object_id]]
        node.extras["rwm"] = rwm.node_extras(
            obj.object_id,
            obj.category,
            obj.parts,
            obj.mass,
            obj.is_static,
            semantics=semantics_by_object_id.get(obj.object_id),
        )

    gltf.scenes[0].extras["rwm"] = rwm.scene_extras(
        scene.seed,
        scene.scene_version,
        scene.dt,
        scene.gravity,
        articulations=[articulation_to_extras(a) for a in scene.articulations],
    )

    accumulator.finalize(gltf)

    # --- extensionsUsed ---
    extensions_used = [rwm.EXT_STATE_SERIES]
    if physics_doc.shapes:
        extensions_used.append(khr_physics.EXT_IMPLICIT_SHAPES)
    if physics_doc.physics_materials or physics_doc.node_physics:
        extensions_used.append(khr_physics.EXT_RIGID_BODIES)
    if scene.lights:
        extensions_used.append(EXT_LIGHTS_PUNCTUAL)
    gltf.extensionsUsed = sorted(set(extensions_used))
    gltf.extensionsRequired = []

    return gltf


# --- decode -------------------------------------------------------------------


def _decode_object_nodes(gltf: pygltflib.GLTF2) -> list[int]:
    """Node indices carrying ``extras.rwm.object_id``, ordered by node index
    (== original encode order, since object nodes are always added first)."""
    indices = []
    for i, node in enumerate(gltf.nodes):
        extras = node.extras or {}
        if "rwm" in extras and "object_id" in extras["rwm"]:
            indices.append(i)
    return indices


def _decode_camera_node(gltf: pygltflib.GLTF2) -> pygltflib.Node:
    for node in gltf.nodes:
        if node.camera is not None:
            return node
    raise ValueError("no camera node found in glTF document")


def _decode_light_nodes(gltf: pygltflib.GLTF2) -> list[tuple[pygltflib.Node, dict]]:
    lights_ext = gltf.extensions.get(EXT_LIGHTS_PUNCTUAL, {})
    light_dicts = lights_ext.get("lights", [])
    out = []
    for node in gltf.nodes:
        extensions = node.extensions or {}
        light_ref = extensions.get(EXT_LIGHTS_PUNCTUAL)
        if light_ref is not None:
            out.append((node, light_dicts[light_ref["light"]]))
    return out


def _decode_object_spec(
    gltf: pygltflib.GLTF2,
    node_index: int,
    physics_doc: khr_physics.KhrPhysicsDocument,
) -> ObjectSpec:
    node = gltf.nodes[node_index]
    node_extras = node.extras["rwm"]

    node_physics = physics_doc.node_physics[node_index]
    shape_index = node_physics["collider"]["geometry"]["shape"]
    material_index = node_physics["collider"]["physicsMaterial"]
    shape_name, size = khr_physics.shape_to_object_size(physics_doc.shapes[shape_index])
    friction, restitution = khr_physics.material_to_friction_restitution(
        physics_doc.physics_materials[material_index]
    )

    mesh = gltf.meshes[node.mesh]
    material = gltf.materials[mesh.primitives[0].material]
    pbr = material.pbrMetallicRoughness
    color = np.array(pbr.baseColorFactor, dtype=np.float32)
    roughness = float(np.float32(pbr.roughnessFactor))
    metallic = float(np.float32(pbr.metallicFactor))

    return ObjectSpec(
        object_id=node_extras["object_id"],
        shape=shape_name,
        size=size,
        color=color,
        roughness=roughness,
        metallic=metallic,
        mass=node_extras["mass"],
        friction=friction,
        restitution=restitution,
        is_static=node_extras["is_static"],
        category=node_extras["category"],
        parts=node_extras.get("parts", {}),
    )


def _decode_camera_spec(gltf: pygltflib.GLTF2) -> CameraSpec:
    node = _decode_camera_node(gltf)
    camera = gltf.cameras[node.camera]
    perspective = camera.perspective
    return CameraSpec(
        position=np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float32),
        rotation=np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        yfov=perspective.yfov,
        znear=perspective.znear,
        zfar=perspective.zfar,
        aspect=perspective.aspectRatio,
    )


def _decode_light_specs(gltf: pygltflib.GLTF2) -> list[LightSpec]:
    specs = []
    for node, light_dict in _decode_light_nodes(gltf):
        light_type = light_dict["type"]
        color = np.array(light_dict["color"], dtype=np.float32)
        intensity = light_dict["intensity"]
        if light_type == "directional":
            specs.append(
                LightSpec(
                    type=light_type,
                    color=color,
                    intensity=intensity,
                    rotation=np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                )
            )
        else:
            specs.append(
                LightSpec(
                    type=light_type,
                    color=color,
                    intensity=intensity,
                    position=np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float32),
                )
            )
    return specs


def _decode_poses(gltf: pygltflib.GLTF2, object_node_indices: list[int], num_frames: int) -> np.ndarray:
    animation = gltf.animations[0]
    n = len(object_node_indices)
    poses = np.zeros((num_frames, n, 7), dtype=np.float32)

    node_to_obj_position = {node_index: i for i, node_index in enumerate(object_node_indices)}

    for channel in animation.channels:
        node_index = channel.target.node
        if node_index not in node_to_obj_position:
            continue
        obj_position = node_to_obj_position[node_index]
        sampler = animation.samplers[channel.sampler]
        data = read_accessor(gltf, sampler.output)
        if channel.target.path == "translation":
            poses[:, obj_position, 0:3] = data
        elif channel.target.path == "rotation":
            poses[:, obj_position, 3:7] = data

    return poses


def episode_from_gltf(gltf: pygltflib.GLTF2) -> Episode:
    """Decode a ``pygltflib.GLTF2`` document (produced by ``episode_to_gltf``) back into an Episode."""
    scene_index = gltf.scene if gltf.scene is not None else 0
    gltf_scene = gltf.scenes[scene_index]
    scene_extras = (gltf_scene.extras or {}).get("rwm", {})

    object_node_indices = _decode_object_nodes(gltf)
    physics_doc = khr_physics.read_khr_physics(gltf)

    objects = [_decode_object_spec(gltf, node_index, physics_doc) for node_index in object_node_indices]
    object_ids_in_order = [obj.object_id for obj in objects]
    node_index_by_object_id = {obj.object_id: node_index for obj, node_index in zip(objects, object_node_indices)}

    camera = _decode_camera_spec(gltf)
    lights = _decode_light_specs(gltf)

    gravity = np.array(scene_extras.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float32)
    articulations = [articulation_from_extras(d) for d in scene_extras.get("articulations", [])]
    scene_state = SceneState(
        objects=objects,
        camera=camera,
        lights=lights,
        gravity=gravity,
        dt=scene_extras.get("dt", 0.0),
        seed=scene_extras.get("seed", 0),
        scene_version=scene_extras.get("scene_version", "wm-scenes-v1"),
        articulations=articulations,
    )

    rwm_doc = rwm.read_state_series(gltf)
    times = read_accessor(gltf, rwm_doc["timesAccessor"])
    num_frames = times.shape[0]

    poses = _decode_poses(gltf, object_node_indices, num_frames)
    extra_channels = rwm.decode_channels(
        gltf, rwm_doc, node_index_by_object_id, object_ids_in_order, num_frames
    )

    series = StateSeries(
        times=times,
        poses=poses,
        lin_vel=extra_channels.get("lin_vel"),
        ang_vel=extra_channels.get("ang_vel"),
        actions=extra_channels.get("actions"),
        pose_var=extra_channels.get("pose_var"),
        joint_pos=extra_channels.get("joint_pos"),
    )

    return Episode(scene=scene_state, series=series)


# --- file I/O -------------------------------------------------------------------


def save_episode(ep: Episode, path: str | Path) -> None:
    """Save ``ep`` as GLB (``.glb``) or, for debugging, glTF+bin (``.gltf``)."""
    path = Path(path)
    gltf = episode_to_gltf(ep)
    if path.suffix.lower() == ".glb":
        gltf.save_binary(str(path))
    else:
        gltf.save(str(path), asset=gltf.asset)


def load_episode(path: str | Path) -> Episode:
    """Load an Episode from a GLB (``.glb``) or glTF+bin (``.gltf``) file."""
    gltf = pygltflib.GLTF2.load(str(path))
    return episode_from_gltf(gltf)
