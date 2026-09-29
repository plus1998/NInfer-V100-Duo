from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path


def image_uri(color: str, size: int = 384) -> str:
    foreground = (255, 0, 0) if color == "red" else (0, 0, 255)
    pixels = bytearray()
    for row in range(size):
        for column in range(size):
            if color == "red":
                inside = (row - size // 2) ** 2 + (column - size // 2) ** 2 < (size // 3) ** 2
            else:
                inside = size // 4 <= row < 3 * size // 4 and size // 4 <= column < 3 * size // 4
            pixels.extend(foreground if inside else (255, 255, 255))
    image = f"P6\n{size} {size}\n255\n".encode() + pixels
    return "data:image/x-portable-pixmap;base64," + base64.b64encode(image).decode()


def request(
    base_url: str, model: str, images: list[str], prompt: str, max_tokens: int,
    barrier: threading.Barrier | None = None,
) -> dict:
    content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "presence_penalty": 0,
        "seed": 1,
        "max_tokens": max_tokens,
        "stream": True,
    }
    http_request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.monotonic()
    first_content = None
    output = []
    finish_reason = None
    with urllib.request.urlopen(http_request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise AssertionError(event["error"])
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content", "")
                if text:
                    first_content = first_content or time.monotonic()
                    output.append(text)
                finish_reason = choice.get("finish_reason") or finish_reason
    finished = time.monotonic()
    assert first_content is not None and finish_reason in {"stop", "length"}
    return {
        "text": "".join(output), "first_content": first_content, "finished": finished,
        "ttft_seconds": first_content - started, "total_seconds": finished - started,
        "finish_reason": finish_reason,
    }


def expect_scene(result: dict, color: str) -> None:
    text = result["text"].lower()
    shape = "circle" if color == "red" else "square"
    assert color in text and shape in text and "white" in text, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Real TP2 Vision and concurrent HTTP smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.request_log.read_text().splitlines()]
    startup = next(record for record in reversed(records) if record["event"] == "server_start")
    assert startup["engine"]["tp"] == 2 and startup["engine"]["vision"]
    assert startup["engine"]["max_concurrency"] >= 2
    assert startup["engine"]["prefill_chunk"] == 1024
    instance = startup["server_instance_id"]
    initial_record_count = len(records)
    images = {color: image_uri(color) for color in ("red", "blue")}
    results = {}
    for color in images:
        result = request(args.base_url, args.model, [images[color]],
                         "Identify the shape, its color, and the background color in one sentence.", 64)
        expect_scene(result, color)
        results[color] = result
    large = request(args.base_url, args.model, [image_uri("blue", 1280)],
                    "Identify the shape, its color, and the background color in one sentence.", 64)
    expect_scene(large, "blue")
    results["cross_chunk"] = large
    multiple = request(args.base_url, args.model, list(images.values()),
                       "Describe both images, identifying each shape, its color, and background.", 128)
    expect_scene(multiple, "red")
    expect_scene(multiple, "blue")
    results["two_images"] = multiple
    barrier = threading.Barrier(2)
    prompt = (
        "First identify the shape, its color and background color in this image. "
        "Then write a detailed 600-word explanation of its visual geometry and contrast."
    )
    concurrent_started_ms = int(time.time() * 1000)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            color: executor.submit(request, args.base_url, args.model, [image], prompt, 384, barrier)
            for color, image in images.items()
        }
        concurrent_results = {color: future.result() for color, future in futures.items()}
    for color, result in concurrent_results.items():
        expect_scene(result, color)
    assert max(result["first_content"] for result in concurrent_results.values()) < min(
        result["finished"] for result in concurrent_results.values()
    ), "the two streamed generation windows did not overlap"
    time.sleep(1)
    records = [json.loads(line) for line in args.request_log.read_text().splitlines()]
    new_records = [record for record in records[initial_record_count:]
                   if record["server_instance_id"] == instance]
    assert not any(record["event"] in {"request_error", "request_rejected"} for record in new_records)
    done = [record for record in new_records if record["event"] == "request_done"]
    assert len(done) == 6, f"expected six completed requests, got {len(done)}"
    assert all(record["timings_seconds"]["vision"] > 0 for record in done)
    assert done[2]["result"]["computed_prefill_tokens"] > 1024
    batches = [record["decode_batch"]["average_size"] for record in new_records
               if record["event"] == "throughput"
               and record["timestamp_unix_ms"] >= concurrent_started_ms
               and record["decode_batch"]["average_size"] is not None]
    assert batches and max(batches) > 1, "no actual multi-row decode batch was observed"
    results["concurrent"] = concurrent_results
    results["max_average_decode_batch"] = max(batches)
    results["requests"] = [
        {"result": record["result"], "timings_seconds": record["timings_seconds"],
         "speculative": record["speculative"]} for record in done
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
