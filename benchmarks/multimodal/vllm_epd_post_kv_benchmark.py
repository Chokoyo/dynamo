# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from dynamo.runtime import DistributedRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument(
        "--requests",
        "--requests-per-level",
        dest="requests_per_level",
        type=int,
        default=64,
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--audit-dir")
    parser.add_argument("--json-output")
    return parser.parse_args()


def parse_concurrency(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("--concurrency must contain positive integers")
    return levels


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round(quantile * (len(ordered) - 1)))
    return ordered[index]


def build_token_ids(tokenizer, variant: str) -> list[int]:
    prompt = (
        f"<|im_start|>user\n{variant} "
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(image_token_id, int) or token_ids.count(image_token_id) != 1:
        raise RuntimeError("benchmark prompt must contain exactly one image token")
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
            raise RuntimeError("vLLM PD worker returned no chunks")
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
    image_url: str,
    pd_worker_id: int,
    encode_worker_id: int,
    source_kind: str,
    cache_salt: str | None = None,
) -> dict[str, Any]:
    cache_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    request = {
        "token_ids": token_ids,
        "sampling_options": {"temperature": 0.0},
        "stop_conditions": {"max_tokens": 8},
        "multi_modal_data": {"image_url": [{"Url": image_url}]},
        "multi_modal_uuids": {"image_url": [cache_key]},
        "mm_routing_info": {
            "epd_prefill_selection": {
                "mode": "enforce",
                "worker_id": pd_worker_id,
                "dp_rank": None,
            },
            "epd_routing_plan": {
                "target_p_worker_id": pd_worker_id,
                "target_p_generation": pd_worker_id,
                "objects": [
                    {
                        "object_index": 0,
                        "source_kind": source_kind,
                        "source_worker_id": encode_worker_id,
                        "source_worker_generation": encode_worker_id,
                        "embedding_cache_key": cache_key,
                        "estimated_cost_ms": 0.0,
                    }
                ],
                "predicted_benefit_ms": 0.0,
                "score_components": {},
            },
        },
    }
    if cache_salt is not None:
        request["nvext"] = {"cache_salt": cache_salt}
    return request


async def run_requests(
    client,
    worker_id: int,
    requests,
    timeout: float,
    concurrency: int,
):
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(request):
        async with semaphore:
            started = time.perf_counter()
            await collect(client, request, worker_id, timeout)
            return (time.perf_counter() - started) * 1_000

    started = time.perf_counter()
    latencies = await asyncio.gather(*(run_one(request) for request in requests))
    elapsed = time.perf_counter() - started
    if not latencies:
        raise RuntimeError("benchmark requires at least one request")
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "throughput_rps": len(latencies) / elapsed,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
    }


def read_audit_events(audit_dir: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if not audit_dir.is_dir():
        return events
    for path in audit_dir.glob("*.json"):
        events[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return events


def summarize_audit_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event"))
        counts[name] = counts.get(name, 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "kv_covered": sum(
            int(event.get("kv_covered", 0))
            for event in events
            if event.get("event") == "CONNECTOR_NO_FETCH"
        ),
        "p_cache_covered": sum(
            int(event.get("p_cache_covered", 0))
            for event in events
            if event.get("event") == "CONNECTOR_NO_FETCH"
        ),
    }


async def run_audited_requests(
    client,
    worker_id: int,
    requests,
    timeout: float,
    audit_dir: Path,
    concurrency: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = read_audit_events(audit_dir)
    metrics = await run_requests(client, worker_id, requests, timeout, concurrency)
    after = read_audit_events(audit_dir)
    new_events = [after[name] for name in sorted(after.keys() - before.keys())]
    return metrics, summarize_audit_events(new_events)


def require_count(audit: dict[str, Any], event: str, expected: int, phase: str) -> None:
    actual = int(audit["counts"].get(event, 0))
    if actual != expected:
        raise RuntimeError(
            f"{phase}: expected {expected} {event} events, observed {actual}: {audit}"
        )


async def run_output_parity_gate(
    *,
    client,
    worker_id: int,
    tokenizer,
    image_url: str,
    encode_worker_id: int,
    timeout: float,
    audit_dir: Path,
) -> dict[str, Any]:
    separator = "&" if "?" in image_url else "?"
    parity_image_url = f"{image_url}{separator}dynamo_epd_output_parity=1"
    token_ids = build_token_ids(tokenizer, "output-parity")
    phases = (
        (
            "cold_e_compute",
            build_request(
                token_ids=token_ids,
                image_url=parity_image_url,
                pd_worker_id=worker_id,
                encode_worker_id=encode_worker_id,
                source_kind="E_COMPUTE",
                cache_salt="epd-output-parity-cold",
            ),
        ),
        (
            "p_local_ec",
            build_request(
                token_ids=token_ids,
                image_url=parity_image_url,
                pd_worker_id=worker_id,
                encode_worker_id=encode_worker_id,
                source_kind="E_CACHE",
                cache_salt="epd-output-parity-p-local",
            ),
        ),
        (
            "full_kv_hit",
            build_request(
                token_ids=token_ids,
                image_url=parity_image_url,
                pd_worker_id=worker_id,
                encode_worker_id=encode_worker_id,
                source_kind="E_CACHE",
                cache_salt="epd-output-parity-p-local",
            ),
        ),
    )
    outputs: dict[str, list[int]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for phase, request in phases:
        before = read_audit_events(audit_dir)
        chunks = await collect(client, request, worker_id, timeout)
        after = read_audit_events(audit_dir)
        outputs[phase] = output_token_ids(chunks)
        audits[phase] = summarize_audit_events(
            [after[name] for name in sorted(after.keys() - before.keys())]
        )

    require_count(audits["cold_e_compute"], "ENCODE_COMPUTE", 1, "parity_cold")
    require_count(
        audits["cold_e_compute"],
        "CONNECTOR_FETCH_REQUESTED",
        1,
        "parity_cold",
    )
    require_count(audits["p_local_ec"], "ENCODE_COMPUTE", 0, "parity_p_local")
    require_count(audits["p_local_ec"], "CONNECTOR_NO_FETCH", 1, "parity_p_local")
    if int(audits["p_local_ec"]["p_cache_covered"]) != 1:
        raise RuntimeError(f"parity_p_local was not covered by P cache: {audits}")
    require_count(audits["full_kv_hit"], "ENCODE_COMPUTE", 0, "parity_kv")
    require_count(audits["full_kv_hit"], "CONNECTOR_NO_FETCH", 1, "parity_kv")
    if int(audits["full_kv_hit"]["kv_covered"]) != 1:
        raise RuntimeError(f"parity_kv was not covered by KV: {audits}")

    expected = outputs["cold_e_compute"]
    mismatches = {
        phase: token_ids
        for phase, token_ids in outputs.items()
        if token_ids != expected
    }
    if mismatches:
        raise RuntimeError(
            f"vLLM EPD output token parity failed: expected={expected}, "
            f"mismatches={mismatches}"
        )
    return {"passed": True, "output_token_ids": expected, "audits": audits}


async def main() -> None:
    args = parse_args()
    concurrency_levels = parse_concurrency(args.concurrency)
    if not args.audit_dir:
        raise RuntimeError("--audit-dir is required for semantic EPD validation")
    audit_dir = Path(args.audit_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    try:
        encode_client, pd_client = await asyncio.gather(
            connect(runtime, f"{args.namespace}.encode.generate", args.timeout),
            connect(runtime, f"{args.namespace}.backend.generate", args.timeout),
        )
        encode_worker_id = min(encode_client.instance_ids())
        pd_worker_id = min(pd_client.instance_ids())

        stable_warm = build_request(
            token_ids=build_token_ids(tokenizer, "p-local-warm"),
            image_url=args.image_url,
            pd_worker_id=pd_worker_id,
            encode_worker_id=encode_worker_id,
            source_kind="E_COMPUTE",
        )
        stable_warm_metrics, stable_warm_audit = await run_audited_requests(
            pd_client, pd_worker_id, [stable_warm], args.timeout, audit_dir, 1
        )

        kv_request = build_request(
            token_ids=build_token_ids(tokenizer, "kv-hit"),
            image_url=args.image_url,
            pd_worker_id=pd_worker_id,
            encode_worker_id=encode_worker_id,
            source_kind="E_CACHE",
        )
        kv_warm_metrics, kv_warm_audit = await run_audited_requests(
            pd_client, pd_worker_id, [kv_request], args.timeout, audit_dir, 1
        )

        require_count(stable_warm_audit, "ENCODE_COMPUTE", 1, "stable_warm")
        require_count(stable_warm_audit, "CONNECTOR_FETCH_REQUESTED", 1, "stable_warm")
        require_count(kv_warm_audit, "ENCODE_COMPUTE", 0, "kv_warm")
        require_count(kv_warm_audit, "CONNECTOR_NO_FETCH", 1, "kv_warm")

        output_parity = await run_output_parity_gate(
            client=pd_client,
            worker_id=pd_worker_id,
            tokenizer=tokenizer,
            image_url=args.image_url,
            encode_worker_id=encode_worker_id,
            timeout=args.timeout,
            audit_dir=audit_dir,
        )

        results = []
        for concurrency in concurrency_levels:
            request_count = max(args.requests_per_level, concurrency)
            cold_requests = [
                build_request(
                    token_ids=build_token_ids(
                        tokenizer, f"cold-c{concurrency}-{index}"
                    ),
                    image_url=(
                        f"{args.image_url}{'&' if '?' in args.image_url else '?'}"
                        f"cold-c{concurrency}-{index}"
                    ),
                    pd_worker_id=pd_worker_id,
                    encode_worker_id=encode_worker_id,
                    source_kind="E_COMPUTE",
                )
                for index in range(request_count)
            ]
            p_local_requests = [
                build_request(
                    token_ids=build_token_ids(
                        tokenizer, f"p-local-c{concurrency}-{index}"
                    ),
                    image_url=args.image_url,
                    pd_worker_id=pd_worker_id,
                    encode_worker_id=encode_worker_id,
                    source_kind="E_CACHE",
                )
                for index in range(request_count)
            ]
            kv_requests = [
                build_request(
                    token_ids=build_token_ids(tokenizer, "kv-hit"),
                    image_url=args.image_url,
                    pd_worker_id=pd_worker_id,
                    encode_worker_id=encode_worker_id,
                    source_kind="E_CACHE",
                )
                for _ in range(request_count)
            ]

            cold_metrics, cold_audit = await run_audited_requests(
                pd_client,
                pd_worker_id,
                cold_requests,
                args.timeout,
                audit_dir,
                concurrency,
            )
            p_local_metrics, p_local_audit = await run_audited_requests(
                pd_client,
                pd_worker_id,
                p_local_requests,
                args.timeout,
                audit_dir,
                concurrency,
            )
            kv_metrics, kv_audit = await run_audited_requests(
                pd_client,
                pd_worker_id,
                kv_requests,
                args.timeout,
                audit_dir,
                concurrency,
            )

            phase_prefix = f"c{concurrency}"
            require_count(
                cold_audit, "ENCODE_COMPUTE", request_count, f"{phase_prefix}_cold"
            )
            require_count(
                cold_audit,
                "CONNECTOR_FETCH_REQUESTED",
                request_count,
                f"{phase_prefix}_cold",
            )
            require_count(p_local_audit, "ENCODE_COMPUTE", 0, f"{phase_prefix}_p_local")
            require_count(
                p_local_audit,
                "CONNECTOR_NO_FETCH",
                request_count,
                f"{phase_prefix}_p_local",
            )
            if int(p_local_audit["p_cache_covered"]) != request_count:
                raise RuntimeError(
                    f"{phase_prefix}_p_local was not covered by P cache: "
                    f"{p_local_audit}"
                )
            require_count(kv_audit, "ENCODE_COMPUTE", 0, f"{phase_prefix}_kv")
            require_count(
                kv_audit,
                "CONNECTOR_NO_FETCH",
                request_count,
                f"{phase_prefix}_kv",
            )
            if int(kv_audit["kv_covered"]) != request_count:
                raise RuntimeError(
                    f"{phase_prefix}_kv was not covered by KV: {kv_audit}"
                )

            results.append(
                {
                    "concurrency": concurrency,
                    "phases": {
                        "cold_e_compute": {
                            "metrics": cold_metrics,
                            "audit": cold_audit,
                        },
                        "p_local_ec": {
                            "metrics": p_local_metrics,
                            "audit": p_local_audit,
                        },
                        "full_kv_hit": {
                            "metrics": kv_metrics,
                            "audit": kv_audit,
                        },
                    },
                }
            )

        payload = {
            "benchmark": "vllm_epd_post_kv",
            "model": args.model,
            "image_url": args.image_url,
            "workers": {"encode": encode_worker_id, "pd": pd_worker_id},
            "warmup": {
                "stable": {"metrics": stable_warm_metrics, "audit": stable_warm_audit},
                "kv": {"metrics": kv_warm_metrics, "audit": kv_warm_audit},
            },
            "output_parity": output_parity,
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
