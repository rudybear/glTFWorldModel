"""Low-level helpers over pygltflib.GLTF2 for reading and writing accessors.

pygltflib gives us the JSON/binary-chunk plumbing for glTF/GLB; it does not
give us numpy round-tripping of accessor data. These helpers fill that gap:

- ``read_accessor`` turns an accessor index into a numpy array, honoring
  bufferView byteOffset/byteStride and whether the buffer lives in the GLB
  binary blob or an external/data-uri buffer.
- ``BufferAccumulator`` is the write-side counterpart: append numpy arrays,
  get back accessor indices, and finalize once into a single buffer 0 +
  binary blob suitable for ``GLTF2.save_binary``.

Only the component type / accessor type combinations gltfworld actually
produces or consumes are supported: float32 (SCALAR/VEC3/VEC4), uint32
(SCALAR), uint16 (SCALAR).
"""

from __future__ import annotations

import numpy as np
import pygltflib

# --- component type <-> numpy dtype -----------------------------------------

_DTYPE_BY_COMPONENT_TYPE: dict[int, np.dtype] = {
    pygltflib.BYTE: np.dtype("<i1"),
    pygltflib.UNSIGNED_BYTE: np.dtype("<u1"),
    pygltflib.SHORT: np.dtype("<i2"),
    pygltflib.UNSIGNED_SHORT: np.dtype("<u2"),
    pygltflib.UNSIGNED_INT: np.dtype("<u4"),
    pygltflib.FLOAT: np.dtype("<f4"),
}

_COMPONENT_TYPE_BY_DTYPE: dict[np.dtype, int] = {
    np.dtype("<i1"): pygltflib.BYTE,
    np.dtype("<u1"): pygltflib.UNSIGNED_BYTE,
    np.dtype("<i2"): pygltflib.SHORT,
    np.dtype("<u2"): pygltflib.UNSIGNED_SHORT,
    np.dtype("<u4"): pygltflib.UNSIGNED_INT,
    np.dtype("<f4"): pygltflib.FLOAT,
}

_NUM_COMPONENTS_BY_TYPE: dict[str, int] = {
    pygltflib.SCALAR: 1,
    pygltflib.VEC2: 2,
    pygltflib.VEC3: 3,
    pygltflib.VEC4: 4,
    pygltflib.MAT2: 4,
    pygltflib.MAT3: 9,
    pygltflib.MAT4: 16,
}

_TYPE_BY_NUM_COMPONENTS: dict[int, str] = {
    1: pygltflib.SCALAR,
    2: pygltflib.VEC2,
    3: pygltflib.VEC3,
    4: pygltflib.VEC4,
}

_ALIGNMENT_BYTES = 4


def _dtype_for_component_type(component_type: int) -> np.dtype:
    try:
        return _DTYPE_BY_COMPONENT_TYPE[component_type]
    except KeyError as exc:
        raise ValueError(f"unsupported componentType {component_type}") from exc


def _component_type_for_dtype(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype).newbyteorder("<")
    try:
        return _COMPONENT_TYPE_BY_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported numpy dtype {dtype}") from exc


def _accessor_type_for_num_components(n: int) -> str:
    try:
        return _TYPE_BY_NUM_COMPONENTS[n]
    except KeyError as exc:
        raise ValueError(f"unsupported number of components {n}") from exc


def _buffer_bytes(gltf: pygltflib.GLTF2, buffer_index: int) -> bytes:
    buffer = gltf.buffers[buffer_index]
    if buffer.uri is None:
        blob = gltf.binary_blob()
        if blob is None:
            raise ValueError("buffer has no uri and no binary blob is set")
        return blob
    return bytes(gltf.get_data_from_buffer_uri(buffer.uri))


def read_accessor(gltf: pygltflib.GLTF2, index: int) -> np.ndarray:
    """Read accessor ``index`` off ``gltf`` into a numpy array.

    Returns shape ``(count,)`` for SCALAR accessors, ``(count, n)`` for
    VECn accessors. Handles bufferView byteOffset/byteStride (interleaved
    data) and buffers backed by the GLB binary blob or a data/file uri.
    """
    accessor = gltf.accessors[index]
    dtype = _dtype_for_component_type(accessor.componentType)
    ncomp = _NUM_COMPONENTS_BY_TYPE[accessor.type]
    count = accessor.count
    elem_size = dtype.itemsize * ncomp

    if accessor.bufferView is None:
        # Sparse-only / zero-filled accessor; not produced or consumed by
        # gltfworld, but return zeros rather than raising so callers that
        # merely enumerate accessors don't explode.
        shape = (count,) if ncomp == 1 else (count, ncomp)
        return np.zeros(shape, dtype=dtype)

    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer_bytes = _buffer_bytes(gltf, buffer_view.buffer)

    base_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = buffer_view.byteStride

    if stride is None or stride == elem_size:
        # Tightly packed: one fast fromstring/view.
        raw = buffer_bytes[base_offset : base_offset + count * elem_size]
        flat = np.frombuffer(raw, dtype=dtype, count=count * ncomp)
        arr = flat.reshape(count, ncomp) if ncomp > 1 else flat.reshape(count)
    else:
        # Interleaved: gather element-by-element.
        arr = np.empty((count, ncomp), dtype=dtype)
        for i in range(count):
            start = base_offset + i * stride
            raw = buffer_bytes[start : start + elem_size]
            arr[i] = np.frombuffer(raw, dtype=dtype, count=ncomp)
        if ncomp == 1:
            arr = arr.reshape(count)

    return np.ascontiguousarray(arr)


class BufferAccumulator:
    """Accumulates numpy arrays into a single glTF buffer 0 + binary blob.

    Usage::

        acc = BufferAccumulator()
        idx = acc.add_accessor(gltf, positions, type_="VEC3", is_position=True)
        ...
        acc.finalize(gltf)
    """

    def __init__(self) -> None:
        self._data = bytearray()

    def _pad_to_alignment(self) -> None:
        remainder = len(self._data) % _ALIGNMENT_BYTES
        if remainder:
            self._data.extend(b"\x00" * (_ALIGNMENT_BYTES - remainder))

    def add_accessor(
        self,
        gltf: pygltflib.GLTF2,
        array: np.ndarray,
        *,
        type_: str | None = None,
        target: int | None = None,
        normalized: bool = False,
        compute_minmax: bool = False,
        name: str | None = None,
    ) -> int:
        """Append ``array`` and register a bufferView + accessor for it.

        ``array`` must be a numpy array with a supported dtype, shape
        ``(count,)`` (SCALAR) or ``(count, n)`` (VECn). ``type_`` is
        inferred from the shape if omitted. Returns the new accessor index.
        """
        array = np.ascontiguousarray(array)
        if array.ndim == 1:
            ncomp = 1
        elif array.ndim == 2:
            ncomp = array.shape[1]
        else:
            raise ValueError(f"unsupported array shape {array.shape}")

        inferred_type = _accessor_type_for_num_components(ncomp)
        if type_ is not None and type_ != inferred_type:
            raise ValueError(f"type_={type_!r} does not match array shape {array.shape}")
        accessor_type = type_ or inferred_type

        component_type = _component_type_for_dtype(array.dtype)
        count = array.shape[0]

        self._pad_to_alignment()
        byte_offset = len(self._data)
        self._data.extend(array.tobytes())

        buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=byte_offset,
            byteLength=array.nbytes,
            target=target,
        )
        gltf.bufferViews.append(buffer_view)
        buffer_view_index = len(gltf.bufferViews) - 1

        min_ = None
        max_ = None
        if compute_minmax and count > 0:
            if ncomp == 1:
                min_ = [float(array.min())]
                max_ = [float(array.max())]
            else:
                min_ = [float(v) for v in array.min(axis=0)]
                max_ = [float(v) for v in array.max(axis=0)]

        accessor = pygltflib.Accessor(
            bufferView=buffer_view_index,
            byteOffset=0,
            componentType=component_type,
            normalized=normalized,
            count=count,
            type=accessor_type,
            min=min_,
            max=max_,
            name=name,
        )
        gltf.accessors.append(accessor)
        return len(gltf.accessors) - 1

    def finalize(self, gltf: pygltflib.GLTF2) -> None:
        """Write accumulated bytes into ``gltf``'s buffer 0 + binary blob."""
        self._pad_to_alignment()
        data = bytes(self._data)
        buffer = pygltflib.Buffer(byteLength=len(data))
        gltf.buffers = [buffer]
        gltf.set_binary_blob(data)
