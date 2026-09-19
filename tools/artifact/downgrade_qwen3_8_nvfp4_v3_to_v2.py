#!/usr/bin/env python3
"""Reframe the official Qwen3.8 NVFP4 v3 artifact for the V100 Duo v2 loader.

The payload is reflink-cloned and remains byte-for-byte unchanged. Only the entry
directory is replaced: v3 logical bindings are folded back into the exact fused
objects registered by the current qwen3.8-27b/nvfp4 package.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import struct


V3_MAGIC = b"NINFER\x00\x03"
V2_MAGIC = b"NINFER\x00\x02"
HEADER_V3_BYTES = 32
HEADER_V2_BYTES = 16
ALIGNMENT = 4096
FICLONE = 0x40049409

FORMATS = {
    "bf16": "BF16",
    "fp32": "FP32",
    "int32": "I32",
    "q4_g64_fp16": "Q4G64_F16S",
    "q5_g64_fp16": "Q5G64_F16S",
    "q6_g64_fp16": "Q6G64_F16S",
    "q8_g32_fp16": "W8G32_F16S",
    "nvfp4": "NVFP4",
    "fp8_e4m3fn_row_bf16": "FP8_E4M3FN_ROW_BF16S",
}
LAYOUTS = {
    "contiguous_le_v1": "contiguous-le-v1",
    "row_split_k128_v1": "row-split-k128-v1",
    "block_scale_k16_m128x4_v1": "blockscale-k16-m128x4-v1",
    "row_scale_v1": "row-scale-v1",
}


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_directory(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        header = stream.read(HEADER_V3_BYTES)
        if len(header) != HEADER_V3_BYTES or header[:8] != V3_MAGIC:
            raise ValueError(f"{path}: expected a NInfer v3 entry")
        json_bytes = struct.unpack_from("<Q", header, 8)[0]
        raw = stream.read(json_bytes)
    directory = json.loads(raw)
    payload_start = align_up(HEADER_V3_BYTES + json_bytes)
    if directory.get("metadata", {}).get("name") != "qwen3.8-27b":
        raise ValueError("artifact is not the registered qwen3.8-27b package")
    if len(directory.get("files", [])) != 1:
        raise ValueError("compatibility conversion requires a single-file artifact")
    return directory, payload_start


class ContractBuilder:
    def __init__(self, directory: dict, source: Path, payload_start: int):
        self.directory = directory
        self.source = source
        self.payload_start = payload_start
        self.objects = {value["id"]: value for value in directory["objects"]}
        self.bindings = directory["bindings"]
        self.uses_by_parameter: dict[str, list[dict]] = {}
        for use in directory["uses"]:
            self.uses_by_parameter.setdefault(use["parameter"], []).append(use)
        self.selected: dict[str, str] = {}
        self.physical_names: dict[str, str] = {}

    def binding_parts(self, logical: str) -> list[dict]:
        try:
            binding = self.bindings[logical]
        except KeyError as error:
            raise ValueError(f"missing v3 binding {logical}") from error
        if "object" in binding:
            obj = self.objects[binding["object"]]
            return [{"object": binding["object"], "range": [0, math.prod(obj["shape"])]}]
        return binding["parts"]

    def add(self, old_name: str, *logical_names: str) -> str:
        parts = [part for logical in logical_names for part in self.binding_parts(logical)]
        object_ids = {part["object"] for part in parts}
        if len(object_ids) != 1:
            raise ValueError(f"{old_name}: fused binding spans multiple physical objects")
        object_id = object_ids.pop()
        obj = self.objects[object_id]
        ranges = sorted(tuple(part["range"]) for part in parts)
        cursor = 0
        for begin, end in ranges:
            if begin != cursor or end <= begin:
                raise ValueError(f"{old_name}: binding ranges do not form one packed object")
            cursor = end
        if cursor != math.prod(obj["shape"]):
            raise ValueError(f"{old_name}: binding coverage is incomplete")
        self._select(old_name, object_id)
        return object_id

    def _select(self, old_name: str, object_id: str) -> None:
        if old_name in self.selected:
            raise ValueError(f"duplicate v2 name {old_name}")
        previous = self.physical_names.get(object_id)
        if previous is not None:
            raise ValueError(f"physical object {object_id} mapped as both {previous} and {old_name}")
        self.selected[old_name] = object_id
        self.physical_names[object_id] = old_name

    def add_input_divisor(self, old_name: str, *logical_names: str) -> None:
        object_ids = []
        values = []
        with self.source.open("rb") as stream:
            for logical in logical_names:
                candidates = {
                    use.get("auxiliaries", {}).get("activation_input_divisor", {}).get("object")
                    for use in self.uses_by_parameter.get(logical, [])
                }
                candidates.discard(None)
                if len(candidates) != 1:
                    raise ValueError(f"{logical}: expected one activation input divisor")
                object_id = candidates.pop()
                obj = self.objects[object_id]
                if obj["format"] != "fp32" or obj["layout"] != "contiguous_le_v1" or obj["shape"]:
                    raise ValueError(f"{logical}: invalid activation input divisor")
                stream.seek(self.payload_start + obj["offset"])
                value = stream.read(obj["bytes"])
                object_ids.append(object_id)
                values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"{old_name}: fused parameters have different input divisors")
        self._select(old_name, object_ids[0])

    def add_resource(self, old_name: str, component: str, role: str) -> None:
        object_id = self.directory["components"][component]["resources"][role]
        self._select(old_name, object_id)

    def v2_objects(self) -> list[dict]:
        result = []
        for old_name, object_id in self.selected.items():
            source = self.objects[object_id]
            value = {
                "name": old_name,
                "kind": source["kind"],
                "offset": source["offset"],
                "bytes": source["bytes"],
            }
            if source["kind"] == "tensor":
                value.update(
                    shape=source["shape"],
                    format=FORMATS[source["format"]],
                    layout=LAYOUTS[source["layout"]],
                )
            elif source["kind"] == "resource" and source["encoding"] == "raw_bytes_v1":
                value["encoding"] = "raw-bytes-v1"
            else:
                raise ValueError(f"{object_id}: unsupported object representation")
            result.append(value)
        result.sort(key=lambda value: value["offset"])
        cursor = 0
        for value in result:
            if value["offset"] < cursor:
                raise ValueError(f"{value['name']}: selected payload ranges overlap")
            cursor = value["offset"] + value["bytes"]
        return result


def embedded_chat_template(directory: dict, source: Path, payload_start: int) -> bytes:
    objects = {value["id"]: value for value in directory["objects"]}
    object_id = directory["components"]["text"]["resources"]["tokenizer_config.json"]
    obj = objects[object_id]
    with source.open("rb") as stream:
        stream.seek(payload_start + obj["offset"])
        config = json.loads(stream.read(obj["bytes"]))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer_config.json has no scalar chat_template")
    return template.encode("utf-8")


def build_contract(directory: dict, source: Path, payload_start: int) -> tuple[dict, list[tuple[int, bytes]]]:
    out = ContractBuilder(directory, source, payload_start)
    for role, component in (
        ("tokenizer.json", "text"),
        ("tokenizer_config.json", "text"),
        ("chat_template.jinja", "text"),
        ("generation_config.json", "text"),
        ("preprocessor_config.json", "vision"),
        ("video_preprocessor_config.json", "vision"),
    ):
        out.add_resource("frontend/" + role, component, role)

    for name in ("text/token_embedding", "text/output_head", "text/final_norm"):
        out.add(name, name)
    out.add("text/draft_head", "proposal/head")
    out.add("text/draft_head_token_ids", "proposal/token_ids")

    full_layers = set(range(3, 64, 4))
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        out.add(prefix + "input_norm", prefix + "input_norm")
        if layer in full_layers:
            out.add(
                prefix + "attention/query_key_gate_value",
                *(prefix + "attention/" + role for role in ("query", "key", "gate", "value")),
            )
            for role in ("query_norm", "key_norm", "output"):
                out.add(prefix + "attention/" + role, prefix + "attention/" + role)
        else:
            for role in ("a_log", "dt_bias", "convolution"):
                out.add(prefix + "gdn/" + role, prefix + "gdn/" + role)
            out.add(
                prefix + "gdn/a_b_projection",
                prefix + "gdn/a_projection",
                prefix + "gdn/b_projection",
            )
            out.add(
                prefix + "gdn/query_key_value_z",
                *(prefix + "gdn/" + role for role in ("query", "key", "value", "z")),
            )
            for role in ("norm", "output"):
                out.add(prefix + "gdn/" + role, prefix + "gdn/" + role)
        out.add(prefix + "post_attention_norm", prefix + "post_attention_norm")
        gate_up = prefix + "mlp/gate_up"
        gate = prefix + "mlp/gate"
        up = prefix + "mlp/up"
        object_id = out.add(gate_up, gate, up)
        out.add(prefix + "mlp/down", prefix + "mlp/down")
        if out.objects[object_id]["format"] == "nvfp4":
            out.add_input_divisor(prefix + "mlp/gate_up_projection/input_scale_divisor", gate, up)
            out.add_input_divisor(
                prefix + "mlp/down_projection/input_scale_divisor", prefix + "mlp/down"
            )

    for name in ("input_projection", "embedding_norm", "hidden_norm", "final_norm"):
        out.add("mtp/" + name, "mtp/" + name)
    old_mtp = "mtp/layer/"
    new_mtp = "mtp/layers/0/"
    for role in ("input_norm", "post_attention_norm"):
        out.add(old_mtp + role, new_mtp + role)
    out.add(
        old_mtp + "attention/query_key_gate_value",
        *(new_mtp + "attention/" + role for role in ("query", "key", "gate", "value")),
    )
    for role in ("query_norm", "key_norm", "output"):
        out.add(old_mtp + "attention/" + role, new_mtp + "attention/" + role)
    out.add(old_mtp + "mlp/gate_up", new_mtp + "mlp/gate", new_mtp + "mlp/up")
    out.add(old_mtp + "mlp/down", new_mtp + "mlp/down")

    for name in ("patch_embedding", "patch_embedding_bias", "position_embedding"):
        out.add("vision/" + name, "vision/" + name)
    for layer in range(27):
        prefix = f"vision/layers/{layer}/"
        out.add(
            prefix + "attention/qkv",
            *(prefix + "attention/" + role for role in ("query", "key", "value")),
        )
        out.add(
            prefix + "attention/qkv_bias",
            *(prefix + "attention/" + role + "_bias" for role in ("query", "key", "value")),
        )
        for role in ("output", "output_bias"):
            out.add(prefix + "attention/" + role, prefix + "attention/" + role)
        for role in ("fc1", "fc1_bias", "fc2", "fc2_bias"):
            out.add(prefix + "mlp/" + role, prefix + "mlp/" + role)
        for norm in ("norm1", "norm2"):
            for part in ("weight", "bias"):
                out.add(prefix + norm + "/" + part, prefix + norm + "_" + part)
    for role in ("fc1", "fc1_bias", "fc2", "fc2_bias"):
        out.add("vision/merger/" + role, "vision/merger/" + role)
    for part in ("weight", "bias"):
        out.add("vision/merger/norm/" + part, "vision/merger/norm_" + part)

    objects = out.v2_objects()
    chat_template = embedded_chat_template(directory, source, payload_start)
    chat_object = next(value for value in objects if value["name"] == "frontend/chat_template.jinja")
    if len(chat_template) > chat_object["bytes"]:
        raise ValueError("embedded chat template does not fit the standalone resource allocation")
    chat_object["bytes"] = len(chat_template)
    if len(objects) != 1124:
        raise ValueError(f"expected 1124 base v2 objects, constructed {len(objects)}")
    return (
        {
            "identity": {"model_id": "qwen3.8-27b", "weights_id": "nvfp4"},
            "objects": objects,
        },
        [(payload_start + chat_object["offset"], chat_template)],
    )


def clone_and_reframe(source: Path, destination: Path, directory: dict, payload_start: int,
                      overwrites: list[tuple[int, bytes]]) -> None:
    encoded = json.dumps(directory, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    json_bytes = payload_start - HEADER_V2_BYTES
    if len(encoded) > json_bytes:
        raise ValueError("v2 compatibility directory does not fit before the existing payload")
    if destination.exists():
        raise FileExistsError(destination)
    try:
        with source.open("rb") as src, destination.open("xb") as dst:
            try:
                fcntl.ioctl(dst.fileno(), FICLONE, src.fileno())
            except OSError:
                dst.close()
                destination.unlink()
                shutil.copyfile(source, destination)
        with destination.open("r+b", buffering=0) as stream:
            stream.write(V2_MAGIC + struct.pack("<Q", json_bytes))
            stream.write(encoded)
            stream.write(b" " * (json_bytes - len(encoded)))
            for offset, data in overwrites:
                stream.seek(offset)
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    directory, payload_start = read_directory(args.source)
    compatible, overwrites = build_contract(directory, args.source, payload_start)
    clone_and_reframe(args.source, args.destination, compatible, payload_start, overwrites)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "payload_offset": payload_start,
                "objects": len(compatible["objects"]),
                "payload_bytes_preserved": args.source.stat().st_size - payload_start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
