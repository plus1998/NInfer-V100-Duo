"""Preserve the local Qwen3.8-27B Q4_K_M GGUF in a native .ninfer artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
from typing import Iterator

from tools.artifact.container import (
    ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec,
)
from tools.artifact.layouts import align_up


IDENTITY = ArtifactIdentity("qwen3.8-27b", "gguf-q4-k-m")
BLOCK_BYTES = {12: 144, 14: 210}
DIRECT_BYTES = {0: 4, 30: 2}
TOKEN_DOMAIN = 248077
VOCAB_ROWS = 248320
GDN_KEY_HEADS = 16
GDN_VALUE_REPEATS = 3
GDN_HEAD_DIM = 128


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]
    type: int
    offset: int
    bytes: int


class Gguf:
    """Read GGUF v3 metadata and expose source byte spans without repacking weights."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.cursor = 0
        if self.take(4) != b"GGUF" or self.scalar(4) != 3:
            raise ValueError("expected a little-endian GGUF version 3 file")
        tensor_count, field_count = self.scalar(10), self.scalar(10)
        self.metadata = {}
        for _ in range(field_count):
            key = self.string()
            self.metadata[key] = self.scalar(self.scalar(4))
        tensors = []
        for _ in range(tensor_count):
            name = self.string()
            dimensions = tuple(self.scalar(10) for _ in range(self.scalar(4)))
            type_id, offset = self.scalar(4), self.scalar(10)
            elements = math.prod(dimensions)
            if type_id in BLOCK_BYTES:
                if dimensions[0] % 256:
                    raise ValueError(f"{name}: K does not contain whole K256 blocks")
                size = elements // 256 * BLOCK_BYTES[type_id]
            elif type_id in DIRECT_BYTES:
                size = elements * DIRECT_BYTES[type_id]
            else:
                raise ValueError(f"{name}: unregistered source GGML type {type_id}")
            tensors.append(GgufTensor(name, tuple(reversed(dimensions)), type_id, offset, size))
        self.base = align_up(self.cursor, self.metadata.get("general.alignment", 32))
        self.tensors = {tensor.name: tensor for tensor in tensors}
        if len(self.tensors) != tensor_count:
            raise ValueError("duplicate GGUF tensor names")
        for tensor in tensors:
            if self.base + tensor.offset + tensor.bytes > len(self.data):
                raise ValueError(f"{tensor.name}: source payload is truncated")

    def take(self, count: int) -> bytes:
        begin = self.cursor
        self.cursor += count
        if self.cursor > len(self.data):
            raise ValueError("truncated GGUF metadata")
        return self.data[begin:self.cursor]

    def string(self) -> str:
        return self.take(self.scalar(10)).decode("utf-8")

    def scalar(self, type_id: int):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                   7: "?", 10: "Q", 11: "q", 12: "d"}
        if type_id == 8:
            return self.string()
        if type_id == 9:
            item_type, count = self.scalar(4), self.scalar(10)
            return [self.scalar(item_type) for _ in range(count)]
        if type_id not in formats:
            raise ValueError(f"unknown GGUF metadata type {type_id}")
        format = "<" + formats[type_id]
        return struct.unpack(format, self.take(struct.calcsize(format)))[0]

    def payload(self, tensor: GgufTensor) -> memoryview:
        begin = self.base + tensor.offset
        return memoryview(self.data)[begin:begin + tensor.bytes]

    def close(self):
        self.data.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class RowRange:
    tensor: GgufTensor
    begin: int
    end: int


@dataclass(frozen=True)
class QuantObject:
    name: str
    rows: tuple[RowRange, ...]

    @property
    def shape(self) -> tuple[int, int]:
        k = self.rows[0].tensor.shape[1]
        if any(len(row.tensor.shape) != 2 or row.tensor.shape[1] != k or
               row.tensor.type not in BLOCK_BYTES or not 0 <= row.begin < row.end <= row.tensor.shape[0]
               for row in self.rows):
            raise ValueError(f"{self.name}: inconsistent source row range")
        return sum(row.end - row.begin for row in self.rows), k

    @property
    def spec(self) -> TensorSpec:
        n, k = self.shape
        size = align_up(n * 8, 256) + sum(
            (part.end - part.begin) * (k // 256) * BLOCK_BYTES[part.tensor.type]
            for part in self.rows)
        return TensorSpec(self.name, (n, k), "GGML_K", "ggml-k256-v1", size)

    def chunks(self, source: Gguf) -> Iterator[bytes | memoryview]:
        n, k = self.shape
        header = bytearray(align_up(n * 8, 256))
        offset = index = 0
        for part in self.rows:
            row_bytes = k // 256 * BLOCK_BYTES[part.tensor.type]
            tag = int(part.tensor.type == 14)
            for _ in range(part.begin, part.end):
                struct.pack_into("<Q", header, index * 8, (offset << 1) | tag)
                index += 1
                offset += row_bytes
        yield header
        for part in self.rows:
            row_bytes = k // 256 * BLOCK_BYTES[part.tensor.type]
            begin = source.base + part.tensor.offset + part.begin * row_bytes
            end = source.base + part.tensor.offset + part.end * row_bytes
            # Bounded views let the writer stream the large embedding and output head.
            for offset in range(begin, end, 16 * 1024 * 1024):
                yield memoryview(source.data)[offset:min(offset + 16 * 1024 * 1024, end)]


@dataclass(frozen=True)
class DirectObject:
    spec: TensorSpec
    data: bytes


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def frontend_resources(source: Gguf, vision: Gguf,
                       chat_template_override: str | None = None) -> dict[str, bytes]:
    metadata = source.metadata
    tokens, types = metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.token_type"]
    if len(tokens) != VOCAB_ROWS or len(types) != VOCAB_ROWS:
        raise ValueError("source tokenizer must contain 248320 registered rows")
    vocab, added = {}, []
    for index, (text, token_type) in enumerate(zip(tokens[:TOKEN_DOMAIN], types[:TOKEN_DOMAIN])):
        if token_type == 1:
            if text in vocab:
                raise ValueError("source vocabulary contains duplicate token text")
            vocab[text] = index
        else:
            if token_type not in (3, 4):
                raise ValueError(f"token {index} has unsupported type {token_type}")
            added.append(dict(id=index, content=text, single_word=False, lstrip=False,
                              rstrip=False, normalized=False, special=token_type == 3))
    template = chat_template_override or metadata["tokenizer.chat_template"]
    tokenizer = {"model": {"type": "BPE", "vocab": vocab,
                            "merges": metadata["tokenizer.ggml.merges"]}, "added_tokens": added}
    config = {"add_bos_token": False, "add_prefix_space": False,
              "pad_token": tokens[metadata["tokenizer.ggml.padding_token_id"]],
              "chat_template": template,
              "added_tokens_decoder": {str(token["id"]): {k: v for k, v in token.items() if k != "id"}
                                       for token in added}}
    image = {"patch_size": vision.metadata["clip.vision.patch_size"],
             "temporal_patch_size": 2, "merge_size": vision.metadata["clip.vision.spatial_merge_size"],
             "image_mean": vision.metadata["clip.vision.image_mean"],
             "image_std": vision.metadata["clip.vision.image_std"],
             "size": {"shortest_edge": 65536, "longest_edge": 16777216}}
    video = {**image, "size": {"shortest_edge": 65536, "longest_edge": 1048576},
             "fps": 2.0, "min_frames": 4, "max_frames": 768}
    return {"frontend/tokenizer.json": json_bytes(tokenizer),
            "frontend/tokenizer_config.json": json_bytes(config),
            "frontend/chat_template.jinja": template.encode("utf-8"),
            "frontend/generation_config.json": json_bytes({"eos_token_id": metadata["tokenizer.ggml.eos_token_id"]}),
            "frontend/preprocessor_config.json": json_bytes(image),
            "frontend/video_preprocessor_config.json": json_bytes(video)}


def gdn_grouped_source_heads() -> tuple[int, ...]:
    """GGUF tiled [repeat,key] indices in the family's grouped [key,repeat] order."""
    return tuple(repeat * GDN_KEY_HEADS + key
                 for key in range(GDN_KEY_HEADS) for repeat in range(GDN_VALUE_REPEATS))


def direct(source: Gguf, name: str, source_name: str, *, shape=None,
           unit_offset: bool = False, a_log: bool = False, transpose: bool = False,
           gdn_value_head_dim: int | None = None, gdn_prefix_rows: int = 0) -> DirectObject:
    import numpy as np
    tensor = source.tensors[source_name]
    if tensor.type != 0:
        raise ValueError(f"{source_name}: expected FP32 direct source")
    values = np.frombuffer(source.payload(tensor), dtype="<f4").copy().reshape(tensor.shape)
    if gdn_value_head_dim is not None:
        expected_rows = gdn_prefix_rows + GDN_KEY_HEADS * GDN_VALUE_REPEATS * gdn_value_head_dim
        if values.shape[0] != expected_rows:
            raise ValueError(f"{source_name}: unexpected GDN value-head extent")
        order = list(range(gdn_prefix_rows)) + [
            gdn_prefix_rows + head * gdn_value_head_dim + dim
            for head in gdn_grouped_source_heads() for dim in range(gdn_value_head_dim)]
        values = values[order]
    if unit_offset:
        values -= np.float32(1)
    if a_log:
        if not np.all(values < 0):
            raise ValueError("source ssm_a must be finite negative values")
        values = np.log(-values).astype("<f4")
    if transpose:
        values = values.T.copy()
    format = "FP32" if a_log or name.endswith("gdn/dt_bias") else "BF16"
    if format == "BF16":
        words = values.view("<u4")
        payload = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2").tobytes()
    else:
        payload = values.astype("<f4").tobytes()
    return DirectObject(TensorSpec(name, tuple(shape or values.shape), format, "contiguous-le-v1"), payload)


def build_gdn_objects(source: Gguf, prefix: str, source_prefix: str) -> list[QuantObject | DirectObject]:
    """Undo llama.cpp's V-head tiling without changing a quantized block.

    Its _LinearAttentionVReorderBase converts grouped [key,repeat,dim] to tiled
    [repeat,key,dim] for V/Z, controls, state parameters, convolution V channels,
    and output-projection columns. Rows can be restored verbatim; output columns
    remain tiled because a 128-wide head can bisect a Q4_K/Q6_K 256-value block.
    The output projection consumes grouped activations through its tiled-column Op.
    """
    def grouped_rows(suffix: str, head_dim: int, prefix_rows: int = 0):
        tensor = source.tensors[source_prefix + suffix]
        expected = prefix_rows + GDN_KEY_HEADS * GDN_VALUE_REPEATS * head_dim
        if tensor.shape[0] != expected:
            raise ValueError(f"{tensor.name}: unexpected GDN value-head extent")
        result = [RowRange(tensor, 0, prefix_rows)] if prefix_rows else []
        result.extend(RowRange(tensor, prefix_rows + head * head_dim,
                               prefix_rows + (head + 1) * head_dim)
                      for head in gdn_grouped_source_heads())
        return tuple(result)

    qk_rows = 2 * GDN_KEY_HEADS * GDN_HEAD_DIM
    output = source.tensors[source_prefix + "ssm_out.weight"]
    return [
        direct(source, prefix + "gdn/a_log", source_prefix + "ssm_a", a_log=True,
               gdn_value_head_dim=1),
        direct(source, prefix + "gdn/dt_bias", source_prefix + "ssm_dt.bias",
               gdn_value_head_dim=1),
        direct(source, prefix + "gdn/convolution", source_prefix + "ssm_conv1d.weight",
               transpose=True, gdn_value_head_dim=GDN_HEAD_DIM, gdn_prefix_rows=qk_rows),
        QuantObject(prefix + "gdn/a_b_projection",
                    grouped_rows("ssm_alpha.weight", 1) + grouped_rows("ssm_beta.weight", 1)),
        QuantObject(prefix + "gdn/query_key_value_z",
                    grouped_rows("attn_qkv.weight", GDN_HEAD_DIM, qk_rows) +
                    grouped_rows("attn_gate.weight", GDN_HEAD_DIM)),
        direct(source, prefix + "gdn/norm", source_prefix + "ssm_norm.weight"),
        QuantObject(prefix + "gdn/output", (RowRange(output, 0, output.shape[0]),)),
    ]


def build_text_objects(source: Gguf, ranking: Path) -> list[QuantObject | DirectObject]:
    def rows(name: str, begin=0, end=None):
        tensor = source.tensors[name]
        return RowRange(tensor, begin, tensor.shape[0] if end is None else end)

    def quant(name: str, *source_names: str):
        return QuantObject(name, tuple(rows(item) for item in source_names))

    def attention(prefix: str, layer: int):
        p = f"blk.{layer}."
        q = source.tensors[p + "attn_q.weight"]
        query = tuple(RowRange(q, head * 512, head * 512 + 256) for head in range(24))
        gate = tuple(RowRange(q, head * 512 + 256, (head + 1) * 512) for head in range(24))
        return [QuantObject(prefix + "attention/query_key_gate_value",
                            query + (rows(p + "attn_k.weight"),) + gate + (rows(p + "attn_v.weight"),)),
                direct(source, prefix + "attention/query_norm", p + "attn_q_norm.weight", unit_offset=True),
                direct(source, prefix + "attention/key_norm", p + "attn_k_norm.weight", unit_offset=True),
                quant(prefix + "attention/output", p + "attn_output.weight")]

    def mlp(prefix: str, layer: int):
        p = f"blk.{layer}."
        return [quant(prefix + "mlp/gate_up", p + "ffn_gate.weight", p + "ffn_up.weight"),
                quant(prefix + "mlp/down", p + "ffn_down.weight")]

    objects = [quant("text/token_embedding", "token_embd.weight")]
    for layer in range(64):
        prefix, p = f"text/layers/{layer}/", f"blk.{layer}."
        objects.append(direct(source, prefix + "input_norm", p + "attn_norm.weight", unit_offset=True))
        if layer % 4 == 3:
            objects.extend(attention(prefix, layer))
        else:
            objects.extend(build_gdn_objects(source, prefix, p))
        objects.append(direct(source, prefix + "post_attention_norm", p + "post_attention_norm.weight", unit_offset=True))
        objects.extend(mlp(prefix, layer))
    objects.extend([direct(source, "text/final_norm", "output_norm.weight", unit_offset=True),
                    quant("text/output_head", "output.weight")])
    import numpy as np
    counts = np.fromfile(ranking, dtype="<i8")
    if counts.size % VOCAB_ROWS or not counts.size:
        raise ValueError("ranking must contain complete 248320-entry rows")
    counts = counts.reshape(-1, VOCAB_ROWS)[0, :TOKEN_DOMAIN]
    forced = [i for i, t in enumerate(source.metadata["tokenizer.ggml.token_type"][:TOKEN_DOMAIN]) if t == 3]
    ordered = np.argsort(-counts, kind="stable").tolist()
    forced_set = set(forced)
    shortlist = [i for i in ordered if i not in forced_set][:131072 - len(forced)] + forced
    shortlist.sort(key=lambda i: -int(counts[i]))
    objects.extend([QuantObject("text/draft_head", tuple(rows("output.weight", i, i + 1) for i in shortlist)),
                    DirectObject(TensorSpec("text/draft_head_token_ids", (131072,), "I32", "contiguous-le-v1"),
                                 np.asarray(shortlist, dtype="<i4").tobytes()),
                    quant("mtp/input_projection", "blk.64.nextn.eh_proj.weight"),
                    direct(source, "mtp/embedding_norm", "blk.64.nextn.enorm.weight", unit_offset=True),
                    direct(source, "mtp/hidden_norm", "blk.64.nextn.hnorm.weight", unit_offset=True),
                    direct(source, "mtp/layer/input_norm", "blk.64.attn_norm.weight", unit_offset=True)])
    objects.extend(attention("mtp/layer/", 64))
    objects.append(direct(source, "mtp/layer/post_attention_norm", "blk.64.post_attention_norm.weight", unit_offset=True))
    objects.extend(mlp("mtp/layer/", 64))
    objects.append(direct(source, "mtp/final_norm", "blk.64.nextn.shared_head_norm.weight", unit_offset=True))
    return objects


def preflight(source: Gguf, vision: Gguf):
    expected = {"general.architecture": "qwen35", "general.file_type": 15,
                "qwen35.block_count": 65, "qwen35.embedding_length": 5120,
                "qwen35.feed_forward_length": 17408, "qwen35.attention.head_count": 24,
                "qwen35.attention.head_count_kv": 4, "qwen35.nextn_predict_layers": 1,
                "qwen35.ssm.group_count": GDN_KEY_HEADS,
                "qwen35.ssm.time_step_rank": GDN_KEY_HEADS * GDN_VALUE_REPEATS,
                "qwen35.ssm.state_size": GDN_HEAD_DIM,
                "qwen35.ssm.inner_size": GDN_KEY_HEADS * GDN_VALUE_REPEATS * GDN_HEAD_DIM,
                "qwen35.full_attention_interval": 4, "qwen35.rope.dimension_sections": [11, 11, 10, 0]}
    for key, value in expected.items():
        if source.metadata.get(key) != value:
            raise ValueError(f"source {key} does not match Qwen3.8-27B Q4_K_M")
    if Counter(t.type for t in source.tensors.values()) != {0: 360, 12: 439, 14: 67}:
        raise ValueError("source is not the registered lmstudio-community Q4_K_M inventory")
    if len(vision.tensors) != 334 or vision.metadata.get("general.architecture") != "clip":
        raise ValueError("expected the companion Qwen3.8-27B BF16 Vision GGUF")


def convert(model: Path, mmproj: Path, output: Path, ranking: Path,
            inspect_only=False, chat_template_override: str | None = None):
    with Gguf(model) as source, Gguf(mmproj) as vision:
        preflight(source, vision)
        resources = frontend_resources(source, vision, chat_template_override)
        objects = build_text_objects(source, ranking)
        specs = [ResourceSpec(name, "raw-bytes-v1", len(data)) for name, data in resources.items()]
        specs.extend(obj.spec for obj in objects)
        vision_specs = [TensorSpec("vision/gguf/" + tensor.name, tensor.shape,
                                   "BF16" if tensor.type == 30 else "FP32", "contiguous-le-v1")
                        for tensor in vision.tensors.values()]
        specs.extend(vision_specs)
        report = {"identity": {"model_id": IDENTITY.model_id, "weights_id": IDENTITY.weights_id},
                  "source": str(model.resolve()), "mmproj": str(mmproj.resolve()),
                  "tensor_count": len(objects) + len(vision_specs), "resource_count": len(resources),
                  "source_formats": dict(Counter(t.type for t in source.tensors.values())),
                  "resource_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in resources.items()},
                  "transforms": ["Q4_K and Q6_K block bytes preserved after row fusion and selection",
                                 "unit-offset normalization source gains minus one, round to BF16",
                                 "A_log = FP32 log(-GGUF ssm_a); original exponential rounding is not invertible",
                                 "convolution transposed to tap-major and rounded to BF16",
                                 "GGUF token metadata reconstructed into six native frontend resources",
                                 "Vision GGUF tensors retained unchanged; Vision execution disabled"]}
        if inspect_only:
            print(json.dumps(report, indent=2))
            return report
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, IDENTITY, specs) as writer:
            for name, data in resources.items():
                writer.write(name, data)
            for index, obj in enumerate(objects):
                writer.write(obj.spec.name, obj.chunks(source) if isinstance(obj, QuantObject) else obj.data)
                if index % 32 == 0:
                    print(f"[{index + 1}/{len(objects)}] {obj.spec.name}", flush=True)
            for spec, tensor in zip(vision_specs, vision.tensors.values()):
                writer.write(spec.name, vision.payload(tensor))
        report["output_bytes"] = output.stat().st_size
        report_path = output.with_name(output.name + ".conversion.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output} ({report['output_bytes']} bytes)", flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, default=Path(__file__).resolve().parents[3] / "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64")
    parser.add_argument("--chat-template", type=Path, default=None,
                        help="Override chat template with a .jinja file instead of using the GGUF's embedded template")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    chat_template_override = args.chat_template.read_text(encoding="utf-8") if args.chat_template else None
    convert(args.model, args.mmproj, args.out, args.ranking, args.inspect_only, chat_template_override)


if __name__ == "__main__":
    main()
