"""Round-trip tests for gltfworld.gltf.accessors: write via BufferAccumulator,
read back via read_accessor, for every dtype/type combo we use on the wire,
plus alignment edge cases."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from gltfworld.gltf.accessors import BufferAccumulator, read_accessor


def _roundtrip(gltf: pygltflib.GLTF2, array: np.ndarray, **kwargs) -> np.ndarray:
    acc = BufferAccumulator()
    index = acc.add_accessor(gltf, array, **kwargs)
    acc.finalize(gltf)
    return read_accessor(gltf, index)


@pytest.mark.parametrize(
    ("dtype", "type_", "shape"),
    [
        (np.float32, pygltflib.SCALAR, (7,)),
        (np.float32, pygltflib.VEC3, (5, 3)),
        (np.float32, pygltflib.VEC4, (5, 4)),
        (np.uint32, pygltflib.SCALAR, (9,)),
        (np.uint16, pygltflib.SCALAR, (9,)),
    ],
)
def test_roundtrip_dtype_type_combo(dtype, type_, shape):
    gltf = pygltflib.GLTF2()
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        array = rng.uniform(-100, 100, size=shape).astype(dtype)
    else:
        array = rng.integers(0, 1000, size=shape).astype(dtype)

    result = _roundtrip(gltf, array, type_=type_)
    assert result.dtype == array.dtype
    assert result.shape == array.shape
    np.testing.assert_array_equal(result, array)


def test_odd_length_uint16_index_buffer():
    """An odd-count uint16 SCALAR accessor (e.g. a triangle-list index buffer
    with a count not divisible by 4/8) exercises the 4-byte alignment padding
    logic: 3 * 2-byte elements = 6 bytes, which needs 2 bytes of padding
    before the *next* accessor to stay 4-byte aligned."""
    gltf = pygltflib.GLTF2()
    indices = np.array([0, 1, 2, 3, 4], dtype=np.uint16)  # 5 elements: odd, not 4-aligned in bytes (10 bytes)

    acc = BufferAccumulator()
    idx1 = acc.add_accessor(gltf, indices, type_=pygltflib.SCALAR, target=pygltflib.ELEMENT_ARRAY_BUFFER)
    # A second accessor right after: if alignment padding is wrong, this will
    # either misread or fail a real glTF-Validator alignment check.
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    idx2 = acc.add_accessor(gltf, positions, type_=pygltflib.VEC3, compute_minmax=True)
    acc.finalize(gltf)

    # Second accessor's bufferView must start on a 4-byte boundary.
    bv2 = gltf.bufferViews[gltf.accessors[idx2].bufferView]
    assert bv2.byteOffset % 4 == 0

    result1 = read_accessor(gltf, idx1)
    result2 = read_accessor(gltf, idx2)
    np.testing.assert_array_equal(result1, indices)
    np.testing.assert_array_equal(result2, positions)


def test_buffer_total_length_is_4byte_aligned():
    gltf = pygltflib.GLTF2()
    acc = BufferAccumulator()
    acc.add_accessor(gltf, np.array([1, 2, 3], dtype=np.uint16), type_=pygltflib.SCALAR)
    acc.finalize(gltf)
    assert gltf.buffers[0].byteLength % 4 == 0
    assert len(gltf.binary_blob()) == gltf.buffers[0].byteLength


def test_position_minmax_computed():
    gltf = pygltflib.GLTF2()
    positions = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    acc = BufferAccumulator()
    index = acc.add_accessor(gltf, positions, type_=pygltflib.VEC3, compute_minmax=True)
    acc.finalize(gltf)

    accessor = gltf.accessors[index]
    assert accessor.min == [1.0, -2.0, -6.0]
    assert accessor.max == [4.0, 5.0, 3.0]


def test_no_minmax_when_not_requested():
    gltf = pygltflib.GLTF2()
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    acc = BufferAccumulator()
    index = acc.add_accessor(gltf, positions, type_=pygltflib.VEC3)
    acc.finalize(gltf)
    assert gltf.accessors[index].min is None
    assert gltf.accessors[index].max is None


def test_multiple_accessors_share_one_buffer():
    gltf = pygltflib.GLTF2()
    acc = BufferAccumulator()
    a = np.arange(10, dtype=np.float32)
    b = np.arange(20, dtype=np.uint32)
    idx_a = acc.add_accessor(gltf, a, type_=pygltflib.SCALAR)
    idx_b = acc.add_accessor(gltf, b, type_=pygltflib.SCALAR)
    acc.finalize(gltf)

    assert len(gltf.buffers) == 1
    np.testing.assert_array_equal(read_accessor(gltf, idx_a), a)
    np.testing.assert_array_equal(read_accessor(gltf, idx_b), b)


def test_scalar_accessor_shape_is_1d():
    gltf = pygltflib.GLTF2()
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _roundtrip(gltf, values, type_=pygltflib.SCALAR)
    assert result.ndim == 1
