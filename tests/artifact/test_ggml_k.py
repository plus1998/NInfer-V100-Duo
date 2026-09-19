"""Exact preservation tests for the source GGUF row transform and mixed-row container."""

import struct
from types import SimpleNamespace

import pytest

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import validate_ggml_k_payload
from tools.convert.qwen3_8_27b.convert_gguf import (
    GgufTensor, QuantObject, RowRange, build_gdn_objects,
)


def source_and_object():
    q4_rows = [bytes((row * 37 + byte) % 256 for byte in range(288)) for row in range(3)]
    q6_rows = [bytes((row * 71 + byte) % 256 for byte in range(420)) for row in range(2)]
    source = SimpleNamespace(base=0, data=b"".join(q4_rows + q6_rows))
    q4 = GgufTensor("q4", (3, 512), 12, 0, 864)
    q6 = GgufTensor("q6", (2, 512), 14, 864, 840)
    obj = QuantObject("mixed", (RowRange(q4, 2, 3), RowRange(q6, 0, 2), RowRange(q4, 0, 1)))
    return source, obj, [q4_rows[2], *q6_rows, q4_rows[0]]


def test_fusion_preserves_all_source_block_bytes(tmp_path):
    source, obj, expected_rows = source_and_object()
    payload = b"".join(obj.chunks(source))
    assert struct.unpack_from("<4Q", payload) == (0, 577, 1417, 2256)
    assert payload[32:256] == bytes(224)
    assert payload[256:] == b"".join(expected_rows)
    validate_ggml_k_payload((4, 512), payload)
    path = tmp_path / "mixed.ninfer"
    with ArtifactWriter(path, ArtifactIdentity("fixture", "ggml-k"), [obj.spec]) as writer:
        writer.write("mixed", payload)
    with Artifact(path) as artifact:
        assert artifact.find("mixed").shape == (4, 512)
        assert bytes(artifact.payload("mixed")) == payload


@pytest.mark.parametrize("offset,value", [(0, 2), (8, 0), (32, 1)])
def test_rejects_corrupt_descriptor_or_padding(offset, value):
    source, obj, _ = source_and_object()
    payload = bytearray(b"".join(obj.chunks(source)))
    payload[offset] = value
    with pytest.raises(ValueError):
        validate_ggml_k_payload((4, 512), payload)


def test_gdn_restores_grouped_heads_without_changing_quantized_blocks():
    import numpy as np

    # Independent source encoder: llama.cpp transposes the HF [key,repeat,dim]
    # head axes to [repeat,key,dim]. Build that GGUF representation and require
    # the converter's complete GDN inventory to recover the original row bytes.
    tensors, parts, grouped = {}, [], {}

    def tiled(values, head_dim, prefix=0):
        tail = values[prefix:].reshape(16, 3, head_dim, *values.shape[1:])
        axes = (1, 0, *range(2, tail.ndim))
        return np.concatenate((values[:prefix], tail.transpose(axes).reshape(values[prefix:].shape)))

    def add(name, values, type_id, shape):
        encoded = values.tobytes()
        full_name = "blk.0." + name
        tensors[full_name] = GgufTensor(full_name, shape, type_id,
                                       sum(len(part) for part in parts), len(encoded))
        parts.append(encoded)

    def quant(name, rows, type_id, head_dim=None, prefix=0, columns=256):
        row_bytes = (144 if type_id == 12 else 210) * (columns // 256)
        values = np.zeros((rows, row_bytes), dtype=np.uint8)
        # Every row carries a unique exact marker, including heads separated by
        # multiples of 256 rows, which a one-byte periodic filler would alias.
        values[:, :4] = np.arange(rows, dtype="<u4").view(np.uint8).reshape(rows, 4)
        values[:, 4:] = np.arange(row_bytes - 4, dtype=np.uint32).astype(np.uint8)
        grouped[name] = values
        add(name, tiled(values, head_dim, prefix) if head_dim else values,
            type_id, (rows, columns))

    quant("attn_qkv.weight", 10240, 14, 128, 4096)
    quant("attn_gate.weight", 6144, 12, 128)
    quant("ssm_alpha.weight", 48, 12, 1)
    quant("ssm_beta.weight", 48, 14, 1)
    quant("ssm_out.weight", 5, 12, columns=6144)

    a = -(np.arange(48, dtype="<f4") + 1)
    dt = np.arange(48, dtype="<f4") / 8
    conv = (np.arange(10240, dtype="<f4")[:, None] // 128 +
            np.arange(4, dtype="<f4")[None, :] / 2)
    norm = np.ones(128, dtype="<f4")
    add("ssm_a", tiled(a, 1), 0, a.shape)
    add("ssm_dt.bias", tiled(dt, 1), 0, dt.shape)
    add("ssm_conv1d.weight", tiled(conv, 128, 4096), 0, conv.shape)
    add("ssm_norm.weight", norm, 0, norm.shape)
    source = SimpleNamespace(base=0, data=b"".join(parts), tensors=tensors)
    source.payload = lambda tensor: source.data[tensor.offset:tensor.offset + tensor.bytes]
    objects = {obj.spec.name.removeprefix("text/layers/0/gdn/"): obj
               for obj in build_gdn_objects(source, "text/layers/0/", "blk.0.")}

    expected_quant = {
        "query_key_value_z": ("attn_qkv.weight", "attn_gate.weight"),
        "a_b_projection": ("ssm_alpha.weight", "ssm_beta.weight"),
        "output": ("ssm_out.weight",),
    }
    for name, sources in expected_quant.items():
        obj = objects[name]
        payload = b"".join(obj.chunks(source))
        validate_ggml_k_payload(obj.shape, payload)
        prefix_bytes = ((obj.shape[0] * 8 + 255) // 256) * 256
        assert payload[prefix_bytes:] == b"".join(grouped[key].tobytes() for key in sources)
        tags = [int(tensors["blk.0." + key].type == 14)
                for key in sources for _ in range(grouped[key].shape[0])]
        assert [struct.unpack_from("<Q", payload, row * 8)[0] & 1
                for row in range(obj.shape[0])] == tags

    # FP32 state parameters and BF16 convolution are checked against represented
    # grouped inputs; output columns deliberately retain the original GGUF bytes.
    np.testing.assert_array_equal(np.frombuffer(objects["dt_bias"].data, dtype="<f4"), dt)
    np.testing.assert_allclose(np.frombuffer(objects["a_log"].data, dtype="<f4"),
                               np.log(-a.astype(np.float64)), rtol=1e-7, atol=0)
    assert objects["convolution"].spec.shape == (4, 10240)
    assert np.all(conv.view("<u4") & 0xffff == 0)
    assert objects["convolution"].data == (conv.T.copy().view("<u4") >> 16).astype("<u2").tobytes()
    assert objects["norm"].data == (norm.view("<u4") >> 16).astype("<u2").tobytes()
