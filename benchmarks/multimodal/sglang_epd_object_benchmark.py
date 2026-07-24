# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from dynamo.runtime import DistributedRuntime
from dynamo.sglang.multimodal_epd import sglang_image_cache_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--image-url",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    )
    parser.add_argument("--image-count", type=int, default=1)
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--source-kind",
        choices=("E_COMPUTE", "E_CACHE", "P_LOCAL"),
        default="E_COMPUTE",
    )
    parser.add_argument(
        "--source-kinds",
        help=(
            "comma-separated per-object source kinds; overrides --source-kind and "
            "must match --image-count"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--cache-bust-e-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-output")
    return parser.parse_args()


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))
    return ordered[index]


def build_token_ids(model: str, image_count: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    image_token = "<|image_pad|>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    vision_parts = "".join(
        "<|vision_start|><|image_pad|><|vision_end|>" for _ in range(image_count)
    )
    prompt = (
        "<|im_start|>user\n"
        f"{vision_parts}"
        "Describe the provided image(s) in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not isinstance(image_token_id, int)
        or token_ids.count(image_token_id) != image_count
    ):
        raise RuntimeError(
            f"benchmark prompt must contain exactly {image_count} image tokens"
        )
    return token_ids


async def connect(runtime: DistributedRuntime, endpoint_name: str, timeout: float):
    client = await runtime.endpoint(endpoint_name).client()
    await asyncio.wait_for(client.wait_for_instances(), timeout=timeout)
    return client


async def collect(client, request: dict[str, Any], worker_id: int, timeout: float):
    async with asyncio.timeout(timeout):
        stream = await client.direct(request, worker_id)
        chunks = []
        async for response in stream:
            if response.is_error():
                raise RuntimeError("; ".join(response.comments() or ["worker error"]))
            data = response.data()
            if isinstance(data, str):
                data = json.loads(data)
            if data is not None:
                chunks.append(data)
        if not chunks:
            raise RuntimeError("decode worker returned no chunks")
        return chunks


def output_token_ids(chunks: list[dict[str, Any]]) -> list[int]:
    token_ids: list[int] = []
    for chunk in chunks:
        chunk_token_ids = chunk.get("token_ids")
        if isinstance(chunk_token_ids, list):
            token_ids.extend(int(token_id) for token_id in chunk_token_ids)
    if not token_ids:
        raise RuntimeError(f"worker response contained no output token IDs: {chunks!r}")
    return token_ids


def build_request(
    *,
    token_ids: list[int],
    image_urls: list[str],
    prefill_worker_id: int,
    encode_worker_ids: list[int],
    source_kinds: list[str],
    source_worker_ids: list[int] | None = None,
) -> dict[str, Any]:
    if not image_urls:
        raise ValueError("at least one image URL is required")
    if len(source_kinds) != len(image_urls):
        raise ValueError("source kinds must cover every image URL")
    if source_worker_ids is not None and len(source_worker_ids) != len(image_urls):
        raise ValueError("source worker IDs must cover every image URL")
    if (
        any(source_kind != "P_LOCAL" for source_kind in source_kinds)
        and not encode_worker_ids
    ):
        raise ValueError("at least one encode worker is required for remote sources")

    objects = []
    for object_index, (image_url, source_kind) in enumerate(
        zip(image_urls, source_kinds, strict=True)
    ):
        source_worker_id = prefill_worker_id
        if source_kind != "P_LOCAL":
            source_worker_id = (
                source_worker_ids[object_index]
                if source_worker_ids is not None
                else encode_worker_ids[object_index % len(encode_worker_ids)]
            )
        objects.append(
            {
                "object_index": object_index,
                "source_kind": source_kind,
                "source_worker_id": source_worker_id,
                "source_worker_generation": source_worker_id,
                "embedding_cache_key": sglang_image_cache_key(image_url),
                "estimated_cost_ms": 0.0,
            }
        )

    return {
        "token_ids": token_ids,
        "sampling_options": {"temperature": 0.0},
        "stop_conditions": {"max_tokens": 8},
        "multi_modal_data": {
            "image_url": [{"Url": image_url} for image_url in image_urls]
        },
        "mm_routing_info": {
            "epd_prefill_selection": {
                "mode": "enforce",
                "worker_id": prefill_worker_id,
                "data_parallel_rank": None,
            },
            "epd_routing_plan": {
                "target_p_worker_id": prefill_worker_id,
                "target_p_generation": prefill_worker_id,
                "objects": objects,
                "predicted_benefit_ms": 0.0,
                "score_components": {},
            },
        },
    }


async def run_level(
    *,
    client,
    request_factory: Callable[[], dict[str, Any]],
    decode_worker_id: int,
    concurrency: int,
    request_count: int,
    warmup: int,
    timeout: float,
) -> dict[str, float | int]:
    for _ in range(warmup):
        await collect(client, request_factory(), decode_worker_id, timeout)

    semaphore = asyncio.Semaphore(concurrency)

    async def one_request() -> tuple[float, list[int]]:
        async with semaphore:
            started = time.perf_counter()
            chunks = await collect(client, request_factory(), decode_worker_id, timeout)
            return (time.perf_counter() - started) * 1_000, output_token_ids(chunks)

    started = time.perf_counter()
    samples = await asyncio.gather(*(one_request() for _ in range(request_count)))
    elapsed = time.perf_counter() - started
    latencies = [sample[0] for sample in samples]
    outputs = [sample[1] for sample in samples]
    expected_output = outputs[0]
    if any(output != expected_output for output in outputs[1:]):
        raise RuntimeError(
            f"SGLang EPD output token parity failed at concurrency {concurrency}: "
            f"outputs={outputs}"
        )
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "throughput_rps": request_count / elapsed,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
        "output_parity_passed": True,
        "output_token_ids": expected_output,
    }


async def main() -> None:
    args = parse_args()
    if args.image_count <= 0:
        raise ValueError("--image-count must be a positive integer")
    concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    if not concurrency_levels or any(value <= 0 for value in concurrency_levels):
        raise ValueError("concurrency levels must be positive integers")

    os.environ.pop("DYN_SYSTEM_PORT", None)
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    try:
        encode_client, prefill_client, decode_client = await asyncio.gather(
            connect(runtime, f"{args.namespace}.encode.generate", args.timeout),
            connect(runtime, f"{args.namespace}.prefill.generate", args.timeout),
            connect(runtime, f"{args.namespace}.backend.generate", args.timeout),
        )
        encode_worker_ids = sorted(encode_client.instance_ids())
        prefill_worker_id = min(prefill_client.instance_ids())
        decode_worker_id = min(decode_client.instance_ids())
        source_kinds = (
            [value.strip() for value in args.source_kinds.split(",")]
            if args.source_kinds
            else [args.source_kind] * args.image_count
        )
        valid_source_kinds = {"E_COMPUTE", "E_CACHE", "P_LOCAL"}
        if len(source_kinds) != args.image_count:
            raise ValueError("--source-kinds must match --image-count")
        if any(value not in valid_source_kinds for value in source_kinds):
            raise ValueError(
                "--source-kinds values must be E_COMPUTE, E_CACHE, or P_LOCAL"
            )
        token_ids = build_token_ids(args.model, args.image_count)
        stable_image_urls = []
        for object_index in range(args.image_count):
            if args.image_count == 1:
                stable_image_urls.append(args.image_url)
                continue
            separator = "&" if "?" in args.image_url else "?"
            stable_image_urls.append(
                f"{args.image_url}{separator}dynamo_epd_object={object_index}"
            )

        planned_source_worker_ids = [
            (
                prefill_worker_id
                if source_kind == "P_LOCAL"
                else encode_worker_ids[object_index % len(encode_worker_ids)]
            )
            for object_index, source_kind in enumerate(source_kinds)
        ]
        warm_objects = [
            (image_url, source_kind, planned_source_worker_ids[object_index])
            for object_index, (image_url, source_kind) in enumerate(
                zip(stable_image_urls, source_kinds, strict=True)
            )
            if source_kind in ("E_CACHE", "P_LOCAL")
        ]
        if warm_objects:
            warm_image_urls = [item[0] for item in warm_objects]
            warm_worker_ids = [
                item[2] if item[1] == "E_CACHE" else encode_worker_ids[0]
                for item in warm_objects
            ]
            warm_request = build_request(
                token_ids=build_token_ids(args.model, len(warm_image_urls)),
                image_urls=warm_image_urls,
                prefill_worker_id=prefill_worker_id,
                encode_worker_ids=encode_worker_ids,
                source_kinds=["E_COMPUTE"] * len(warm_image_urls),
                source_worker_ids=warm_worker_ids,
            )
            await collect(decode_client, warm_request, decode_worker_id, args.timeout)

        request_counter = itertools.count()

        def request_factory() -> dict[str, Any]:
            image_urls = list(stable_image_urls)
            if "E_COMPUTE" in source_kinds and args.cache_bust_e_compute:
                request_index = next(request_counter)
                image_urls = [
                    (
                        f"{image_url}{'&' if '?' in image_url else '?'}"
                        f"dynamo_epd_benchmark={request_index}"
                        if source_kind == "E_COMPUTE"
                        else image_url
                    )
                    for image_url, source_kind in zip(
                        image_urls, source_kinds, strict=True
                    )
                ]
            return build_request(
                token_ids=token_ids,
                image_urls=image_urls,
                prefill_worker_id=prefill_worker_id,
                encode_worker_ids=encode_worker_ids,
                source_kinds=source_kinds,
                source_worker_ids=planned_source_worker_ids,
            )

        results = [
            await run_level(
                client=decode_client,
                request_factory=request_factory,
                decode_worker_id=decode_worker_id,
                concurrency=concurrency,
                request_count=max(args.requests_per_level, concurrency),
                warmup=args.warmup,
                timeout=args.timeout,
            )
            for concurrency in concurrency_levels
        ]
        payload = {
            "benchmark": "sglang_epd_object_routing",
            "model": args.model,
            "source_kind": source_kinds[0] if len(set(source_kinds)) == 1 else "MIXED",
            "source_kinds": source_kinds,
            "image_count": args.image_count,
            "cache_bust_e_compute": args.cache_bust_e_compute,
            "workers": {
                "encode": encode_worker_ids,
                "prefill": prefill_worker_id,
                "decode": decode_worker_id,
            },
            "results": results,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_output:
            Path(args.json_output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
