#!/usr/bin/env python3
"""Fixed-window OpenAI chat benchmark client for vLLM EPD experiments.

Unlike ``vllm bench serve``, this client sends requests for a fixed wall-clock
window and computes metrics from requests that complete before a bounded settle
period. Throughput/goodput use the configured send window as denominator so
tail drain does not skew comparisons across overload points.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


VLLM_SRC = Path(os.environ.get("VLLM_SRC", "/workspace/vllm-src"))
if str(VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLLM_SRC))

from vllm.benchmarks.datasets import (  # noqa: E402
    RandomDataset,
    RandomMultiModalDataset,
    SampleRequest,
)
from vllm.benchmarks.lib.endpoint_request_func import (  # noqa: E402
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_chat_completions,
)
from vllm.tokenizers import get_tokenizer  # noqa: E402


def parse_metadata(items: list[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid metadata item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, p))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if value == float("inf"):
        return "inf"
    return value


def request_output_len(output: RequestFuncOutput, fallback: int) -> int:
    if output.output_tokens and output.output_tokens > 0:
        return int(output.output_tokens)
    if output.ttft > 0:
        return max(1, len(output.itl) + 1)
    return fallback


def build_request_pool(args: argparse.Namespace) -> list[SampleRequest]:
    tokenizer = get_tokenizer(
        args.tokenizer_model, trust_remote_code=args.trust_remote_code
    )
    pool_size = max(1, args.request_pool_size)
    if args.dataset_name == "random":
        return RandomDataset(random_seed=args.seed).sample(
            tokenizer=tokenizer,
            num_requests=pool_size,
            input_len=args.input_len,
            output_len=args.output_len,
            request_id_prefix=args.request_id_prefix,
        )
    if args.dataset_name == "random-mm":
        total_items = args.images + args.videos
        bucket_config: dict[tuple[int, int, int], float] = {}
        if args.images > 0:
            bucket_config[(args.image_size, args.image_size, 1)] = float(args.images)
        if args.videos > 0:
            bucket_config[
                (args.video_size, args.video_size, args.video_frames)
            ] = float(args.videos)
        return RandomMultiModalDataset(random_seed=args.seed).sample(
            tokenizer=tokenizer,
            num_requests=pool_size,
            input_len=args.input_len,
            output_len=args.output_len,
            base_items_per_request=total_items,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt={"image": args.images, "video": args.videos},
            bucket_config=bucket_config,
            request_id_prefix=args.request_id_prefix,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def delay_for_index(args: argparse.Namespace, rng: np.random.Generator) -> float:
    if args.request_rate == float("inf"):
        return 0.0
    if args.burstiness == float("inf"):
        return 1.0 / args.request_rate
    theta = 1.0 / (args.request_rate * args.burstiness)
    return float(rng.gamma(shape=args.burstiness, scale=theta))


def to_request_input(
    args: argparse.Namespace,
    sample: SampleRequest,
    launch_index: int,
    request_id: str | None = None,
) -> RequestFuncInput:
    stable_request_id = request_id or f"{args.request_id_prefix}{launch_index}"
    return RequestFuncInput(
        model=args.model,
        model_name=args.model,
        prompt=sample.prompt,
        api_url=args.api_url,
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len or args.output_len,
        multi_modal_content=sample.multi_modal_data,
        ignore_eos=True,
        extra_headers={
            "x-request-id": stable_request_id,
            "x-dynamo-meta-replay-request-id": stable_request_id,
        },
        extra_body=args.extra_body,
        request_id=stable_request_id,
    )


async def send_one(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    sample: SampleRequest,
    launch_index: int,
    request_id: str | None = None,
) -> RequestFuncOutput:
    return await async_request_openai_chat_completions(
        request_func_input=to_request_input(args, sample, launch_index, request_id),
        session=session,
        pbar=None,
    )


async def run_warmup(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    pool: list[SampleRequest],
) -> list[Any]:
    if args.num_warmups <= 0:
        return []
    warmups = [
        asyncio.create_task(send_one(args, session, pool[i % len(pool)], -1 - i))
        for i in range(args.num_warmups)
    ]
    return await asyncio.gather(*warmups, return_exceptions=True)


def sample_from_replay(obj: dict[str, Any]) -> SampleRequest:
    return SampleRequest(
        prompt=obj["prompt"],
        prompt_len=int(obj["prompt_len"]),
        expected_output_len=int(obj["output_len"]),
        multi_modal_data=obj.get("multi_modal_content"),
        request_id=obj["sample_id"],
    )


def load_replay_workload(
    path: Path,
    workload_name: str | None,
    stream_name: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, SampleRequest],
]:
    manifest: dict[str, Any] = {}
    workload_defs: dict[str, dict[str, Any]] = {}
    samples: dict[str, SampleRequest] = {}
    requests: list[dict[str, Any]] = []
    seen_workloads: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            row_type = obj.get("type")
            if row_type == "manifest":
                manifest = obj
                for workload in obj.get("workloads", []):
                    workload_defs[workload["name"]] = workload
            elif row_type == "workload":
                workload_defs[obj["name"]] = obj
            elif row_type == "sample":
                sample_id = obj["sample_id"]
                if sample_id in samples:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} at line {lineno}")
                samples[sample_id] = sample_from_replay(obj)
            elif row_type == "request":
                workload = obj.get("workload")
                if not workload:
                    raise ValueError(f"Replay request missing workload at line {lineno}")
                seen_workloads.add(workload)
                if (workload_name is None or workload == workload_name) and (
                    stream_name is None or obj.get("stream") == stream_name
                ):
                    requests.append(obj)
            else:
                raise ValueError(f"Unknown replay row type {row_type!r} at line {lineno}")

    if workload_name is None:
        if len(seen_workloads) != 1:
            raise ValueError(
                "--replay-workload is required when replay JSONL contains "
                f"multiple workloads: {sorted(seen_workloads)}"
            )
        workload_name = next(iter(seen_workloads))
    if not requests:
        suffix = f" stream {stream_name!r}" if stream_name else ""
        raise ValueError(f"No replay requests found for workload {workload_name!r}{suffix}")

    requests.sort(key=lambda row: (float(row["offset_s"]), int(row["request_index"])))
    missing = sorted({row["sample_id"] for row in requests} - set(samples))
    if missing:
        raise ValueError(f"Replay file references missing samples: {missing[:5]}")

    workload = workload_defs.get(workload_name, {"name": workload_name})
    return manifest, workload, requests, samples


async def run_fixed_window(args: argparse.Namespace) -> dict[str, Any]:
    pool = build_request_pool(args)
    connector = aiohttp.TCPConnector(
        limit=args.max_inflight,
        limit_per_host=args.max_inflight,
        ttl_dns_cache=300,
        keepalive_timeout=60,
        force_close=False,
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)

    outputs: list[RequestFuncOutput] = []
    launch_offsets: list[float] = []
    pending: set[asyncio.Task[RequestFuncOutput]] = set()
    rng = np.random.default_rng(args.seed)

    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=timeout,
    ) as session:
        await run_warmup(args, session, pool)

        send_start = time.perf_counter()
        next_launch = send_start
        deadline = send_start + args.duration_seconds
        launch_index = 0
        target_requests = (
            math.ceil(args.duration_seconds * args.request_rate)
            if args.request_rate != float("inf")
            else args.request_pool_size
        )

        def collect_done(done: set[asyncio.Task[RequestFuncOutput]]) -> None:
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = task.result()
                    outputs.append(result)

        while time.perf_counter() < deadline and launch_index < target_requests:
            if len(pending) >= args.max_inflight:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=0.001,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                collect_done(done)
                if len(pending) >= args.max_inflight:
                    await asyncio.sleep(0.001)
                    continue

            now = time.perf_counter()
            if now < next_launch:
                wait_time = min(next_launch - now, 0.05)
                if pending:
                    done, pending = await asyncio.wait(
                        pending,
                        timeout=wait_time,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    collect_done(done)
                else:
                    await asyncio.sleep(wait_time)
                continue

            sample = pool[launch_index % len(pool)]
            task = asyncio.create_task(send_one(args, session, sample, launch_index))
            pending.add(task)
            launch_offsets.append(now - send_start)
            launch_index += 1
            next_launch += delay_for_index(args, rng)

        send_end = time.perf_counter()
        settle_deadline = send_end + args.settle_seconds
        while pending and time.perf_counter() < settle_deadline:
            done, pending = await asyncio.wait(
                pending,
                timeout=min(1.0, max(0.0, settle_deadline - time.perf_counter())),
                return_when=asyncio.FIRST_COMPLETED,
            )
            collect_done(done)

        cancelled = len(pending)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    wall_end = time.perf_counter()
    return summarize(
        args,
        pool,
        outputs,
        launch_offsets,
        target_requests,
        cancelled,
        send_start,
        send_end,
        wall_end,
    )


async def run_replay_window(args: argparse.Namespace) -> dict[str, Any]:
    manifest, workload, replay_requests, samples = load_replay_workload(
        args.replay_jsonl,
        args.replay_workload,
        args.replay_stream,
    )
    pool = list(samples.values())
    duration = float(
        workload.get("duration_seconds")
        or manifest.get("duration_seconds")
        or args.duration_seconds
    )
    args.duration_seconds = duration
    args.request_rate = float(
        len(replay_requests) / duration
        if args.replay_stream is not None and duration > 0
        else workload.get("total_qps")
        or (len(replay_requests) / duration if duration > 0 else 0.0)
    )

    connector = aiohttp.TCPConnector(
        limit=args.max_inflight,
        limit_per_host=args.max_inflight,
        ttl_dns_cache=300,
        keepalive_timeout=60,
        force_close=False,
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)

    outputs: list[RequestFuncOutput] = []
    output_streams: list[str] = []
    launch_offsets: list[float] = []
    launch_streams: list[str] = []
    pending: set[asyncio.Task[tuple[RequestFuncOutput, str]]] = set()

    async def send_replay_one(row: dict[str, Any]) -> tuple[RequestFuncOutput, str]:
        output = await send_one(
            args,
            session,
            samples[row["sample_id"]],
            int(row["request_index"]),
            row.get("request_id"),
        )
        return output, row["stream"]

    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=timeout,
    ) as session:
        warmup_pool = [
            samples[row["sample_id"]]
            for row in replay_requests[: max(1, args.num_warmups)]
        ]
        await run_warmup(args, session, warmup_pool or pool)

        send_start = time.perf_counter()
        send_deadline = send_start + duration
        request_iter = iter(replay_requests)
        next_row = next(request_iter, None)

        def collect_done(done: set[asyncio.Task[tuple[RequestFuncOutput, str]]]) -> None:
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    output, stream = task.result()
                    outputs.append(output)
                    output_streams.append(stream)

        while next_row is not None and time.perf_counter() < send_deadline:
            if len(pending) >= args.max_inflight:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=0.001,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                collect_done(done)
                if len(pending) >= args.max_inflight:
                    if time.perf_counter() >= send_deadline:
                        break
                    await asyncio.sleep(0.001)
                    continue

            target_time = send_start + float(next_row["offset_s"])
            now = time.perf_counter()
            if now >= send_deadline:
                break
            if now < target_time:
                wait_time = min(target_time - now, send_deadline - now, 0.05)
                if pending:
                    done, pending = await asyncio.wait(
                        pending,
                        timeout=wait_time,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    collect_done(done)
                else:
                    await asyncio.sleep(wait_time)
                continue

            pending.add(asyncio.create_task(send_replay_one(next_row)))
            launch_offsets.append(now - send_start)
            launch_streams.append(next_row["stream"])
            next_row = next(request_iter, None)

        send_end = time.perf_counter()
        settle_deadline = send_end + args.settle_seconds
        while pending and time.perf_counter() < settle_deadline:
            done, pending = await asyncio.wait(
                pending,
                timeout=min(1.0, max(0.0, settle_deadline - time.perf_counter())),
                return_when=asyncio.FIRST_COMPLETED,
            )
            collect_done(done)

        cancelled = len(pending)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    wall_end = time.perf_counter()
    result = summarize(
        args,
        pool,
        outputs,
        launch_offsets,
        len(replay_requests),
        cancelled,
        send_start,
        send_end,
        wall_end,
        output_streams=output_streams,
        launch_streams=launch_streams,
    )
    result["replay"] = {
        "jsonl": str(args.replay_jsonl),
        "workload": workload.get("name"),
        "stream": args.replay_stream,
        "unlaunched_requests": len(replay_requests) - len(launch_offsets),
        "manifest": manifest,
        "workload_config": workload,
    }
    return result


async def run_replay_warmup_only(args: argparse.Namespace) -> dict[str, Any]:
    manifest, workload, replay_requests, samples = load_replay_workload(
        args.replay_jsonl,
        args.replay_workload,
        args.replay_stream,
    )
    warmup_pool = [
        samples[row["sample_id"]]
        for row in replay_requests[: max(1, args.num_warmups)]
    ]
    connector = aiohttp.TCPConnector(
        limit=args.max_inflight,
        limit_per_host=args.max_inflight,
        ttl_dns_cache=300,
        keepalive_timeout=60,
        force_close=False,
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=timeout,
    ) as session:
        started = time.perf_counter()
        outputs = await run_warmup(args, session, warmup_pool)
        finished = time.perf_counter()

    successful = [
        output
        for output in outputs
        if isinstance(output, RequestFuncOutput) and output.success
    ]
    failed = [
        output
        for output in outputs
        if isinstance(output, Exception)
        or (isinstance(output, RequestFuncOutput) and not output.success)
    ]
    ttfts = [output.ttft * 1000 for output in successful]
    tpots = []
    for output in successful:
        out_len = request_output_len(output, args.output_len)
        if out_len > 1:
            tpots.append((output.latency - output.ttft) / (out_len - 1) * 1000)
    return {
        "date": dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S"),
        "backend": "openai-chat",
        "endpoint_type": "openai-chat",
        "model_id": args.model,
        "tokenizer_id": args.tokenizer_model,
        "dataset_name": "replay-warmup",
        "target_num_requests": args.num_warmups,
        "launched": args.num_warmups,
        "completed": len(successful),
        "failed": len(failed),
        "cancelled_inflight": 0,
        "duration": finished - started,
        "fixed_window_duration_s": finished - started,
        "send_elapsed_s": finished - started,
        "wall_elapsed_s": finished - started,
        "request_goodput": len(successful) / max(finished - started, 1e-9),
        "p99_ttft_ms": percentile(ttfts, 99),
        "p99_tpot_ms": percentile(tpots, 99),
        "errors": [
            repr(output)
            if isinstance(output, Exception)
            else output.error
            for output in failed
        ],
        "warmup_only": True,
        "replay": {
            "jsonl": str(args.replay_jsonl),
            "workload": workload.get("name"),
            "stream": args.replay_stream,
            "manifest": manifest,
            "workload_config": workload,
        },
    }


def summarize(
    args: argparse.Namespace,
    pool: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    launch_offsets: list[float],
    target_requests: int,
    cancelled: int,
    send_start: float,
    send_end: float,
    wall_end: float,
    output_streams: list[str] | None = None,
    launch_streams: list[str] | None = None,
) -> dict[str, Any]:
    successful = [output for output in outputs if output.success]
    failed = [output for output in outputs if not output.success]
    output_lens = [request_output_len(output, args.output_len) for output in successful]
    ttfts = [output.ttft for output in successful]
    e2els = [output.latency for output in successful]
    tpots = [
        (output.latency - output.ttft) / (out_len - 1)
        for output, out_len in zip(successful, output_lens)
        if out_len > 1
    ]
    itls = [itl for output in successful for itl in output.itl]
    good_completed = 0
    for output, out_len in zip(successful, output_lens):
        tpot = (output.latency - output.ttft) / (out_len - 1) if out_len > 1 else 0.0
        if output.ttft * 1000 <= args.slo_ttft_ms and tpot * 1000 <= args.slo_tpot_ms:
            good_completed += 1

    duration = float(args.duration_seconds)
    total_input = sum(output.prompt_len for output in successful)
    total_output = sum(output_lens)
    result: dict[str, Any] = {
        "date": dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S"),
        "backend": "openai-chat",
        "endpoint_type": "openai-chat",
        "model_id": args.model,
        "tokenizer_id": args.tokenizer_model,
        "dataset_name": args.dataset_name,
        "num_prompts": target_requests,
        "request_pool_size": len(pool),
        "target_num_requests": target_requests,
        "launched": len(launch_offsets),
        "completed": len(successful),
        "failed": len(failed),
        "cancelled_inflight": cancelled,
        "request_rate": args.request_rate,
        "burstiness": "inf" if args.burstiness == float("inf") else args.burstiness,
        "max_concurrency": args.max_inflight,
        "duration": duration,
        "fixed_window_duration_s": duration,
        "send_elapsed_s": send_end - send_start,
        "wall_elapsed_s": wall_end - send_start,
        "settle_seconds": args.settle_seconds,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(successful) / duration,
        "request_goodput": good_completed / duration,
        "fixed_window_request_throughput": len(successful) / duration,
        "fixed_window_request_goodput": good_completed / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "input_lens": [output.prompt_len for output in successful],
        "output_lens": output_lens,
        "ttfts": ttfts,
        "itls": [output.itl for output in successful],
        "e2els": e2els,
        "start_times": [output.start_time for output in successful],
        "launch_offsets": launch_offsets,
        "errors": [output.error for output in failed],
        "generated_texts": [output.generated_text for output in successful],
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
    }

    if output_streams is not None:
        successful_pairs = [
            (output, stream)
            for output, stream in zip(outputs, output_streams)
            if output.success
        ]
        failed_pairs = [
            (output, stream)
            for output, stream in zip(outputs, output_streams)
            if not output.success
        ]
        result["launch_streams"] = launch_streams or []
        result["completed_streams"] = [stream for _, stream in successful_pairs]
        per_stream: dict[str, dict[str, Any]] = {}
        for stream in sorted(set(launch_streams or []) | set(output_streams)):
            stream_successful = [
                output for output, item_stream in successful_pairs if item_stream == stream
            ]
            stream_failed = [
                output for output, item_stream in failed_pairs if item_stream == stream
            ]
            stream_output_lens = [
                request_output_len(output, args.output_len)
                for output in stream_successful
            ]
            stream_ttfts = [output.ttft * 1000 for output in stream_successful]
            stream_e2els = [output.latency * 1000 for output in stream_successful]
            stream_good = 0
            for output, out_len in zip(stream_successful, stream_output_lens):
                tpot = (
                    (output.latency - output.ttft) / (out_len - 1)
                    if out_len > 1
                    else 0.0
                )
                if (
                    output.ttft * 1000 <= args.slo_ttft_ms
                    and tpot * 1000 <= args.slo_tpot_ms
                ):
                    stream_good += 1
            per_stream[stream] = {
                "launched": (launch_streams or []).count(stream),
                "completed": len(stream_successful),
                "failed": len(stream_failed),
                "request_throughput": len(stream_successful) / duration,
                "request_goodput": stream_good / duration,
                "mean_ttft_ms": mean(stream_ttfts),
                "p99_ttft_ms": percentile(stream_ttfts, 99),
                "mean_e2el_ms": mean(stream_e2els),
                "p99_e2el_ms": percentile(stream_e2els, 99),
            }
        result["per_stream"] = per_stream

    for metric, values in {
        "ttft": [v * 1000 for v in ttfts],
        "tpot": [v * 1000 for v in tpots],
        "itl": [v * 1000 for v in itls],
        "e2el": [v * 1000 for v in e2els],
    }.items():
        result[f"mean_{metric}_ms"] = mean(values)
        result[f"median_{metric}_ms"] = median(values)
        result[f"std_{metric}_ms"] = std(values)
        for p in (50, 90, 95, 99):
            result[f"p{p}_{metric}_ms"] = percentile(values, float(p))

    result.update(parse_metadata(args.metadata))
    result["client_args"] = {
        key: jsonable(value)
        for key, value in vars(args).items()
        if key not in {"extra_body"}
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help=(
            "Tokenizer/model id used for prompt accounting. Defaults to --model; "
            "set this when --model is a server-side alias."
        ),
    )
    parser.add_argument("--dataset-name", choices=["random", "random-mm"], default=None)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=150)
    parser.add_argument("--images", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--videos", type=int, default=0)
    parser.add_argument("--video-size", type=int, default=336)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=60.0)
    parser.add_argument("--burstiness", type=float, default=float("inf"))
    parser.add_argument("--max-inflight", type=int, default=4096)
    parser.add_argument("--request-pool-size", type=int, default=512)
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--slo-ttft-ms", type=int, default=20000)
    parser.add_argument("--slo-tpot-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--request-id-prefix", default="fixed-")
    parser.add_argument("--request-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--metadata", nargs="*", default=[])
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--replay-jsonl", type=Path, default=None)
    parser.add_argument("--replay-workload", default=None)
    parser.add_argument("--replay-stream", default=None)
    parser.add_argument("--warmup-only", action="store_true")
    return parser


async def async_main() -> None:
    args = build_parser().parse_args()
    if args.tokenizer_model is None:
        args.tokenizer_model = args.model
    args.api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    args.extra_body = json.loads(args.extra_body_json)
    if args.replay_jsonl is None:
        if args.dataset_name is None:
            raise ValueError("--dataset-name is required without --replay-jsonl")
        if args.input_len is None:
            raise ValueError("--input-len is required without --replay-jsonl")
        if args.request_rate is None or args.request_rate <= 0:
            raise ValueError("--request-rate must be positive")
        if args.dataset_name == "random-mm" and args.images + args.videos <= 0:
            raise ValueError("--images + --videos must be positive for random-mm")
        if args.videos > 0 and args.video_frames <= 1:
            raise ValueError("--video-frames must be greater than 1 for video items")
    else:
        if not args.replay_jsonl.exists():
            raise FileNotFoundError(args.replay_jsonl)
        args.dataset_name = "replay"
        args.input_len = args.input_len or 0
        args.request_rate = args.request_rate or 0.0
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup_only:
        if args.replay_jsonl is None:
            raise ValueError("--warmup-only currently requires --replay-jsonl")
        result = await run_replay_warmup_only(args)
    else:
        result = (
            await run_replay_window(args)
            if args.replay_jsonl is not None
            else await run_fixed_window(args)
        )
    output = args.result_dir / args.result_filename
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("launched", "completed", "failed", "cancelled_inflight", "request_goodput", "p99_ttft_ms", "p99_tpot_ms")}, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
