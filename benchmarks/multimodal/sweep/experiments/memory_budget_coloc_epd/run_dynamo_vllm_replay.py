#!/usr/bin/env python3
"""Run fixed-replay Dynamo+vLLM aggregated and E/PD multimodal experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def default_dynamo_root() -> Path:
    experimental_root = Path("/workspace/dynamo-pr8298-sglang-video-disagg")
    if experimental_root.exists():
        return experimental_root
    return Path("/workspace/dynamo")


DYNAMO_ROOT = Path(os.environ.get("DYNAMO_ROOT", str(default_dynamo_root())))
DYNAMO_RUNTIME_ROOT = Path(os.environ.get("DYNAMO_RUNTIME_ROOT", str(DYNAMO_ROOT)))
VLLM_SRC = Path(os.environ.get("VLLM_SRC", "/workspace/vllm-src"))
FIXED_CLIENT = ROOT / "fixed_rate_client.py"
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_REPLAY = ROOT / "bench" / "text_image_qps32_300s_replay.jsonl"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def now_slug() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def qps_slug(value: str) -> str:
    return value.replace(".", "p")


def cpu_affinity_slug(value: str) -> str:
    return value.replace(",", "-").replace(" ", "")


def thread_env_overrides() -> dict[str, str]:
    return {key: os.environ[key] for key in THREAD_ENV_KEYS if key in os.environ}


def thread_env_suffix() -> str | None:
    overrides = thread_env_overrides()
    if not overrides:
        return None
    values = set(overrides.values())
    if len(values) == 1:
        return f"globalthr{qps_slug(next(iter(values)))}"
    digest = hashlib.sha1(
        json.dumps(overrides, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return f"threadenv{digest}"


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def split_gpu_csv(value: str) -> list[int]:
    gpus = [int(part) for part in split_csv(value)]
    if not gpus:
        raise ValueError("GPU list cannot be empty")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list contains duplicates: {value!r}")
    return gpus


def split_gpu_group_spec(value: str, label: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        group = [int(part.strip()) for part in raw_group.split(",") if part.strip()]
        if not group:
            raise ValueError(f"{label} contains an empty GPU group: {value!r}")
        if len(group) != len(set(group)):
            raise ValueError(
                f"{label} GPU group contains duplicates: {raw_group!r}"
            )
        groups.append(group)
    if not groups:
        raise ValueError(f"{label} cannot be empty")
    return groups


def gpu_groups(gpus: list[int], tp_size: int, label: str) -> list[list[int]]:
    if tp_size < 1:
        raise ValueError(f"{label} tensor parallel size must be >= 1")
    if len(gpus) % tp_size != 0:
        raise ValueError(
            f"{label} GPU count {len(gpus)} must be divisible by "
            f"tensor parallel size {tp_size}: {gpus}"
        )
    return [gpus[i : i + tp_size] for i in range(0, len(gpus), tp_size)]


def encoder_gpu_groups(args: argparse.Namespace, label: str) -> list[list[int]]:
    if args.epd_encoder_gpu_groups:
        groups = split_gpu_group_spec(args.epd_encoder_gpu_groups, label)
        expected = args.epd_encoder_tensor_parallel_size
        bad = [group for group in groups if len(group) != expected]
        if bad:
            raise ValueError(
                f"{label} groups must each have tensor parallel size {expected}: {bad}"
            )
        return groups
    return gpu_groups(
        split_gpu_csv(args.epd_encoder_gpus),
        args.epd_encoder_tensor_parallel_size,
        label,
    )


def flatten_gpu_groups(groups: list[list[int]]) -> list[int]:
    return [gpu for group in groups for gpu in group]


def baseline_gpu_groups(args: argparse.Namespace) -> list[list[int]]:
    return gpu_groups(list(range(4)), args.baseline_tensor_parallel_size, "baseline")


def gpu_group_slug(group: list[int]) -> str:
    return "-".join(str(gpu) for gpu in group)


def has_duplicate_gpu_groups(groups: list[list[int]]) -> bool:
    return len({tuple(group) for group in groups}) < len(groups)


def cuda_visible_devices(group: list[int]) -> str:
    return ",".join(str(gpu) for gpu in group)


def tensor_parallel_args(tp_size: int) -> list[str]:
    if tp_size == 1:
        return []
    return ["--tensor-parallel-size", str(tp_size)]


def epd_worker_gpus(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    encoder_groups = encoder_gpu_groups(args, "EPD encoder")
    encoder_gpus = flatten_gpu_groups(encoder_groups)
    if args.epd_prefill_gpus:
        prefill_gpus = split_gpu_csv(args.epd_prefill_gpus)
    else:
        pd_gpus = split_gpu_csv(args.epd_pd_gpus)
        prefill_gpus = [pd_gpus[0]]

    if args.epd_decode_gpus:
        decode_gpus = split_gpu_csv(args.epd_decode_gpus)
    else:
        pd_gpus = split_gpu_csv(args.epd_pd_gpus)
        decode_gpus = pd_gpus[1:]

    if not prefill_gpus:
        raise ValueError("EPD requires at least one prefill GPU")
    if not decode_gpus:
        raise ValueError("EPD requires at least one decode GPU")
    overlap = sorted(
        (set(encoder_gpus) & set(prefill_gpus))
        | (set(encoder_gpus) & set(decode_gpus))
        | (set(prefill_gpus) & set(decode_gpus))
    )
    if overlap and not args.allow_epd_gpu_overlap:
        raise ValueError(f"EPD GPU roles overlap: {overlap}")
    return encoder_gpus, prefill_gpus, decode_gpus


def e_pd_worker_gpus(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    encoder_gpus = flatten_gpu_groups(encoder_gpu_groups(args, "E/PD encoder"))
    pd_gpus = split_gpu_csv(args.epd_pd_gpus)
    overlap = sorted(set(encoder_gpus) & set(pd_gpus))
    if overlap and not args.allow_epd_gpu_overlap:
        raise ValueError(f"E/PD GPU roles overlap: {overlap}")
    return encoder_gpus, pd_gpus


def hybrid_epd_worker_gpus(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    if args.epd_prefill_gpus or args.epd_decode_gpus:
        pd_gpus = split_gpu_csv(args.epd_pd_gpus)
        prefill_gpus = (
            split_gpu_csv(args.epd_prefill_gpus)
            if args.epd_prefill_gpus
            else [pd_gpus[0]]
        )
        decode_gpus = (
            split_gpu_csv(args.epd_decode_gpus) if args.epd_decode_gpus else pd_gpus[1:]
        )
        if not prefill_gpus:
            raise ValueError("Hybrid EPD requires at least one prefill GPU")
        if not decode_gpus:
            raise ValueError("Hybrid EPD requires at least one decode GPU")
        overlap = sorted(set(prefill_gpus) & set(decode_gpus))
        if overlap:
            raise ValueError(f"Hybrid EPD GPU roles overlap: {overlap}")
        return prefill_gpus, decode_gpus

    all_gpus = split_gpu_csv(args.epd_encoder_gpus) + split_gpu_csv(args.epd_pd_gpus)
    if len(all_gpus) != len(set(all_gpus)) and not args.allow_epd_gpu_overlap:
        raise ValueError(f"Hybrid EPD GPU roles overlap: {sorted(all_gpus)}")
    if len(all_gpus) < 2:
        raise ValueError("Hybrid EPD requires at least two GPUs")
    return [all_gpus[0]], all_gpus[1:]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_replay_workload(path: Path, name: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "workload" and obj.get("name") == name:
                return obj
            if obj.get("type") == "manifest":
                for workload in obj.get("workloads", []):
                    if workload.get("name") == name:
                        return workload
    raise ValueError(f"Replay workload {name!r} not found in {path}")


def read_replay_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "manifest":
                return obj
            break
    return {}


def replay_artifact_workload_slug(args: argparse.Namespace) -> str:
    manifest = read_replay_manifest(args.replay_jsonl)
    image_count = manifest.get("image_count")
    image_size = manifest.get("image_size")
    video_count = manifest.get("video_count", 0)
    video_size = manifest.get("video_size")
    video_frames = manifest.get("video_frames")
    output_len = manifest.get("output_len")
    if output_len is None:
        return args.replay_workload
    if image_count is not None and image_size is not None and not video_count:
        prefix = f"text_image_{image_count}x{image_size}_out{output_len}_"
        if args.replay_workload.startswith("text_image_"):
            return prefix + args.replay_workload.removeprefix("text_image_")
        return prefix + args.replay_workload
    if video_count and video_size is not None and video_frames is not None:
        if image_count:
            prefix = (
                f"text_image_video_{image_count}x{image_size}_"
                f"{video_count}x{video_size}x{video_frames}_out{output_len}_"
            )
            if args.replay_workload.startswith("text_image_video_"):
                return prefix + args.replay_workload.removeprefix("text_image_video_")
            return prefix + args.replay_workload
        prefix = (
            f"text_video_{video_count}x{video_size}x{video_frames}_out{output_len}_"
        )
        if args.replay_workload.startswith("text_video_"):
            return prefix + args.replay_workload.removeprefix("text_video_")
        return prefix + args.replay_workload
    return args.replay_workload


def repeat_slug_from_run_dir(run_dir: Path) -> str | None:
    for part in run_dir.name.split("_"):
        if part.startswith("repeat") and len(part) > len("repeat"):
            return part
    return None


def find_runtime_wheel() -> Path:
    wheels = sorted(
        (DYNAMO_RUNTIME_ROOT / "target" / "wheels").glob("ai_dynamo_runtime-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        raise FileNotFoundError(
            "No ai_dynamo_runtime wheel found. Build one with "
            "`uvx maturin build --release -m <dynamo-root>/lib/bindings/python/Cargo.toml "
            "--out <dynamo-root>/target/wheels`, "
            "or set DYNAMO_RUNTIME_ROOT to a Dynamo checkout with built runtime artifacts."
        )
    return wheels[0]


def ensure_runtime_binding(run_dir: Path) -> Path:
    target = run_dir / "local_wheel"
    marker = target / "dynamo" / "_core.abi3.so"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    wheel = find_runtime_wheel()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
    if not marker.exists():
        raise FileNotFoundError(f"Extracted wheel did not contain {marker}")
    return target


def write_vllm_ms_logging_config(run_dir: Path) -> Path:
    path = run_dir / "logs" / "vllm_logging_ms.json"
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "vllm": {
                "class": "vllm.logging_utils.NewLineFormatter",
                "datefmt": "%m-%d %H:%M:%S",
                "format": (
                    "%(levelname)s %(asctime)s.%(msecs)03d "
                    "[%(fileinfo)s:%(lineno)d] %(message)s"
                ),
            }
        },
        "handlers": {
            "vllm": {
                "class": "logging.StreamHandler",
                "formatter": "vllm",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "vllm": {
                "handlers": ["vllm"],
                "level": "INFO",
                "propagate": False,
            }
        },
    }
    write_json(path, config)
    return path


def base_env(
    args: argparse.Namespace, point_dir: Path, namespace: str
) -> dict[str, str]:
    runtime_path = ensure_runtime_binding(args.run_dir)
    env = os.environ.copy()
    pythonpath = [
        str(runtime_path),
        str(DYNAMO_ROOT / "components" / "src"),
        str(VLLM_SRC),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    ld_paths = [
        str(DYNAMO_RUNTIME_ROOT / "target" / "release"),
        str(DYNAMO_RUNTIME_ROOT / "target" / "release" / "deps"),
        str(DYNAMO_RUNTIME_ROOT / "target" / "debug"),
        str(DYNAMO_RUNTIME_ROOT / "target" / "debug" / "deps"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    env.update(
        {
            "DYN_NAMESPACE": namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "ETCD_ENDPOINTS": f"http://127.0.0.1:{args.etcd_client_port}",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_HTTP_PORT": str(args.http_port),
            "UCX_TLS": args.ucx_tls,
            "UCX_RNDV_SCHEME": "get_zcopy",
            "UCX_RNDV_THRESH": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_TRIM_THRESHOLD_": "0",
            "VLLM_SKIP_P2P_CHECK": "1",
            "HF_HOME": env.get("HF_HOME", "/root/.cache/huggingface"),
            "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": env.get("TRANSFORMERS_OFFLINE", "1"),
            "VLLM_LOGGING_CONFIG_PATH": str(write_vllm_ms_logging_config(point_dir)),
            "VLLM_EPD_LOG_MM_ENCODER_EVENTS": "1",
            "DYNAMO_EPD_LOG_MM_ENCODER_EVENTS": "1",
            "ENABLE_ENCODER_CACHE": "1" if args.enable_encoder_cache else "0",
            "DYN_HYBRID_EPD_DIRECT_TEXT_BYPASS": (
                "1" if args.hybrid_epd_direct_text_bypass else "0"
            ),
        }
    )
    if args.ucx_net_devices:
        env["UCX_NET_DEVICES"] = args.ucx_net_devices
    if args.nixl_write_receiver_buffer_gb is not None:
        env["DYNAMO_NIXL_WRITE_RECEIVER_BUFFER_GB"] = str(
            args.nixl_write_receiver_buffer_gb
        )
    if args.nixl_write_receive_timeout_seconds is not None:
        env["DYNAMO_NIXL_WRITE_RECEIVE_TIMEOUT_SECONDS"] = str(
            args.nixl_write_receive_timeout_seconds
        )
    if args.nixl_write_receiver_device:
        env["DYNAMO_NIXL_WRITE_RECEIVER_DEVICE"] = args.nixl_write_receiver_device
    if args.encoder_max_concurrent is not None:
        env["DYNAMO_EPD_ENCODER_MAX_CONCURRENT"] = str(args.encoder_max_concurrent)
    if args.encode_dispatch_concurrency is not None:
        env["DYN_ENCODE_DISPATCH_CONCURRENCY"] = str(args.encode_dispatch_concurrency)
    if args.encode_batch_size is not None:
        env["DYN_ENCODE_BATCH_SIZE"] = str(args.encode_batch_size)
    if args.eworker_microbatch_max_items is not None:
        env["DYN_EWORKER_MICROBATCH_MAX_ITEMS"] = str(args.eworker_microbatch_max_items)
    if args.eworker_microbatch_wait_ms is not None:
        env["DYN_EWORKER_MICROBATCH_WAIT_MS"] = str(args.eworker_microbatch_wait_ms)
    if args.eworker_parallel_preprocess:
        env["DYN_EWORKER_PARALLEL_PREPROCESS"] = "1"
    if args.eworker_inline_encode:
        env["DYN_EWORKER_INLINE_ENCODE"] = "1"
    if args.eworker_microbatch_thread:
        env["DYN_EWORKER_MICROBATCH_THREAD"] = "1"
    if args.eworker_torch_num_threads is not None:
        env["DYN_EWORKER_TORCH_NUM_THREADS"] = str(args.eworker_torch_num_threads)
    if args.eworker_torch_num_interop_threads is not None:
        env["DYN_EWORKER_TORCH_NUM_INTEROP_THREADS"] = str(
            args.eworker_torch_num_interop_threads
        )
    if args.eworker_timing_breakdown:
        env["DYN_EWORKER_TIMING_BREAKDOWN"] = "1"
    if args.eworker_executor_workers is not None:
        env["DYN_EWORKER_EXECUTOR_WORKERS"] = str(args.eworker_executor_workers)
    if args.eworker_image_decode_executor_workers is not None:
        env["DYN_EWORKER_IMAGE_DECODE_EXECUTOR_WORKERS"] = str(
            args.eworker_image_decode_executor_workers
        )
    if args.mm_torch_num_threads is not None:
        env["DYN_MM_TORCH_NUM_THREADS"] = str(args.mm_torch_num_threads)
    if args.mm_executor_workers is not None:
        env["DYN_MM_EXECUTOR_WORKERS"] = str(args.mm_executor_workers)
    if args.hybrid_epd_prefill_worker_limit is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_LIMIT"] = str(
            args.hybrid_epd_prefill_worker_limit
        )
    if args.hybrid_epd_prefill_worker_burst_limit is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_BURST_LIMIT"] = str(
            args.hybrid_epd_prefill_worker_burst_limit
        )
    if args.hybrid_epd_prefill_worker_burst_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_BURST_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_burst_percent
        )
    if args.hybrid_epd_prefill_worker_dynamic_admission:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_ADMISSION"] = "1"
    if args.hybrid_epd_prefill_worker_dynamic_window is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_WINDOW"] = str(
            args.hybrid_epd_prefill_worker_dynamic_window
        )
    if args.hybrid_epd_prefill_worker_dynamic_target_mm_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_TARGET_MM_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_dynamic_target_mm_percent
        )
    if args.hybrid_epd_prefill_worker_dynamic_gain_per_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_GAIN_PER_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_dynamic_gain_per_percent
        )
    if args.hybrid_epd_prefill_worker_dynamic_min_burst_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_MIN_BURST_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_dynamic_min_burst_percent
        )
    if args.hybrid_epd_prefill_worker_dynamic_max_burst_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_MAX_BURST_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_dynamic_max_burst_percent
        )
    if args.hybrid_epd_prefill_worker_dynamic_smoothing_windows is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_DYNAMIC_SMOOTHING_WINDOWS"] = str(
            args.hybrid_epd_prefill_worker_dynamic_smoothing_windows
        )
    if args.hybrid_epd_prefill_worker_backlog_admission:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_BACKLOG_ADMISSION"] = "1"
    if args.hybrid_epd_prefill_worker_backlog_high_watermark_percent is not None:
        env["DYN_HYBRID_EPD_PREFILL_WORKER_BACKLOG_HIGH_WATERMARK_PERCENT"] = str(
            args.hybrid_epd_prefill_worker_backlog_high_watermark_percent
        )
    if args.hybrid_epd_mm_decode_backlog_guard_max_requests is not None:
        env["DYN_HYBRID_EPD_MM_DECODE_BACKLOG_GUARD_MAX_REQUESTS"] = str(
            args.hybrid_epd_mm_decode_backlog_guard_max_requests
        )
    if args.hybrid_epd_decode_block_pressure_guard_percent is not None:
        env["DYN_HYBRID_EPD_DECODE_BLOCK_PRESSURE_GUARD_PERCENT"] = str(
            args.hybrid_epd_decode_block_pressure_guard_percent
        )
    if args.hybrid_epd_decode_block_pressure_guard_min_available_workers is not None:
        env["DYN_HYBRID_EPD_DECODE_BLOCK_PRESSURE_GUARD_MIN_AVAILABLE_WORKERS"] = str(
            args.hybrid_epd_decode_block_pressure_guard_min_available_workers
        )
    if args.hybrid_epd_decode_request_load_guard_max is not None:
        env["DYN_HYBRID_EPD_DECODE_REQUEST_LOAD_GUARD_MAX"] = str(
            args.hybrid_epd_decode_request_load_guard_max
        )
    if args.hybrid_epd_decode_request_load_guard_min_available_workers is not None:
        env["DYN_HYBRID_EPD_DECODE_REQUEST_LOAD_GUARD_MIN_AVAILABLE_WORKERS"] = str(
            args.hybrid_epd_decode_request_load_guard_min_available_workers
        )
    if args.hybrid_epd_decode_request_load_guard_prefill_worker_limit is not None:
        env["DYN_HYBRID_EPD_DECODE_REQUEST_LOAD_GUARD_PREFILL_WORKER_LIMIT"] = str(
            args.hybrid_epd_decode_request_load_guard_prefill_worker_limit
        )
    if args.hybrid_epd_aux_prefill_decode_request_load_max is not None:
        env["DYN_HYBRID_EPD_AUX_PREFILL_DECODE_REQUEST_LOAD_MAX"] = str(
            args.hybrid_epd_aux_prefill_decode_request_load_max
        )
        env["DYN_HYBRID_EPD_AUX_PREFILL_INCLUDE_UNKNOWN_DECODE_LOAD"] = (
            "1" if args.hybrid_epd_aux_prefill_include_unknown_decode_load else "0"
        )
    if args.hybrid_epd_prefill_backlog_max_tokens is not None:
        env["DYN_HYBRID_EPD_PREFILL_BACKLOG_MAX_TOKENS"] = str(
            args.hybrid_epd_prefill_backlog_max_tokens
        )
    if args.hybrid_epd_prefill_backlog_max_requests is not None:
        env["DYN_HYBRID_EPD_PREFILL_BACKLOG_MAX_REQUESTS"] = str(
            args.hybrid_epd_prefill_backlog_max_requests
        )
    if args.hybrid_epd_reserved_text_decode_workers is not None:
        env["DYN_HYBRID_EPD_RESERVED_TEXT_DECODE_WORKERS"] = str(
            args.hybrid_epd_reserved_text_decode_workers
        )
    if args.hybrid_epd_exclude_reserved_text_decode_from_mm_decode:
        env["DYN_HYBRID_EPD_EXCLUDE_RESERVED_TEXT_DECODE_FROM_MM_DECODE"] = "1"
    if args.hybrid_epd_prefill_overload_spillover:
        env["DYN_HYBRID_EPD_PREFILL_OVERLOAD_SPILLOVER"] = "1"
    if args.runtime_allow_overloaded_dispatch:
        env["DYN_RUNTIME_ALLOW_OVERLOADED_DISPATCH"] = "1"
    return env


def with_env(env: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    merged = env.copy()
    merged.update({k: str(v) for k, v in extra.items()})
    return merged


def with_cpu_affinity(cmd: list[str], cpu_affinity: str | None) -> list[str]:
    if not cpu_affinity:
        return cmd
    return ["taskset", "-c", cpu_affinity, *cmd]


def fpm_env(args: argparse.Namespace, worker_offset: int) -> dict[str, int]:
    if args.forwardpass_metric_port_base is None:
        return {}
    # Each vLLM worker binds DYN_FORWARDPASS_METRIC_PORT + local dp_rank.
    # Reserve a small range per worker so one-process-per-GPU runs do not all
    # collide at the same dp_rank=0 port, and multi-DP workers still have room.
    return {
        "DYN_FORWARDPASS_METRIC_PORT": args.forwardpass_metric_port_base
        + worker_offset * 16
    }


class ManagedProcess:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        log_path: Path,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or ROOT),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()

    def stop(self, timeout: float = 25.0) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=timeout)
        self.log_file.close()


def launch_gpu_sampler(point_dir: Path, topology: str) -> ManagedProcess:
    return ManagedProcess(
        "gpu-sampler",
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        point_dir / "logs" / f"{topology}_replay_gpu_metrics.csv",
        env=os.environ.copy(),
    )


def launch_etcd(args: argparse.Namespace, point_dir: Path) -> ManagedProcess:
    data_dir = point_dir / "etcd-data"
    if data_dir.exists():
        subprocess.run(["rm", "-rf", str(data_dir)], check=True)
    return ManagedProcess(
        "etcd",
        [
            "etcd",
            "--name",
            f"dynamo-{point_dir.name}",
            "--data-dir",
            str(data_dir),
            "--listen-client-urls",
            f"http://127.0.0.1:{args.etcd_client_port}",
            "--advertise-client-urls",
            f"http://127.0.0.1:{args.etcd_client_port}",
            "--listen-peer-urls",
            f"http://127.0.0.1:{args.etcd_peer_port}",
            "--initial-advertise-peer-urls",
            f"http://127.0.0.1:{args.etcd_peer_port}",
            "--initial-cluster",
            f"dynamo-{point_dir.name}=http://127.0.0.1:{args.etcd_peer_port}",
            "--initial-cluster-state",
            "new",
            "--logger",
            "zap",
        ],
        point_dir / "logs" / "etcd.log",
        env=os.environ.copy(),
    )


def url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def url_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return 200 <= resp.status < 500 and (
                '"ready"' in body or '"status"' not in body
            )
    except (urllib.error.URLError, TimeoutError):
        return False


def endpoint_ready(url: str, endpoint: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            if not (200 <= resp.status < 600):
                return False
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    return payload.get("endpoints", {}).get(endpoint) == "ready"


def wait_for(
    label: str,
    predicate,
    processes: Iterable[ManagedProcess],
    timeout_s: int,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for proc in processes:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"{proc.name} exited early with code {code}; see {proc.log_path}"
                )
        if predicate():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {label}")


def common_vllm_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--model",
        args.model,
        "--embedding-transfer-mode",
        args.embedding_transfer_mode,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--mm-processor-cache-gb",
        str(args.mm_processor_cache_gb),
        "--limit-mm-per-prompt",
        json.dumps(
            {"image": args.images_per_request, "video": args.videos_per_request},
            separators=(",", ":"),
        ),
        "--enable-logging-iteration-details",
    ]
    if args.disable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    return cmd


def epd_pd_vllm_args(args: argparse.Namespace) -> list[str]:
    cmd = common_vllm_args(args)
    if args.dynamo_embedding_cache_capacity_gb > 0:
        cmd += [
            "--multimodal-embedding-cache-capacity-gb",
            str(args.dynamo_embedding_cache_capacity_gb),
        ]
    return cmd


def kv_transfer_args(kv_events_port: int) -> list[str]:
    return [
        "--kv-transfer-config",
        json.dumps({"kv_connector": "NixlConnector", "kv_role": "kv_both"}),
        "--kv-events-config",
        json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_events_port}",
            },
            separators=(",", ":"),
        ),
    ]


def launch_frontend(
    args: argparse.Namespace,
    point_dir: Path,
    env: dict[str, str],
    topology: str,
) -> ManagedProcess:
    cmd = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(args.http_port),
        "--router-mode",
        args.frontend_router_mode,
    ]
    if args.frontend_active_decode_blocks_threshold is not None:
        cmd += [
            "--active-decode-blocks-threshold",
            str(args.frontend_active_decode_blocks_threshold),
        ]
    if args.frontend_active_prefill_tokens_threshold is not None:
        cmd += [
            "--active-prefill-tokens-threshold",
            str(args.frontend_active_prefill_tokens_threshold),
        ]
    if args.frontend_active_prefill_tokens_threshold_frac is not None:
        cmd += [
            "--active-prefill-tokens-threshold-frac",
            str(args.frontend_active_prefill_tokens_threshold_frac),
        ]
    return ManagedProcess(
        "frontend",
        with_cpu_affinity(cmd, args.frontend_cpu_affinity),
        point_dir / "logs" / f"{topology}_frontend.log",
        env=env,
        cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
    )


def launch_baseline(
    args: argparse.Namespace, point_dir: Path, env: dict[str, str]
) -> list[ManagedProcess]:
    procs = [launch_frontend(args, point_dir, env, "baseline")]
    groups = baseline_gpu_groups(args)
    for worker_idx, group in enumerate(groups):
        group_slug = gpu_group_slug(group)
        procs.append(
            ManagedProcess(
                f"baseline-agg-gpu{group_slug}",
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--enable-multimodal",
                        "--disaggregation-mode",
                        "agg",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_pd),
                        *tensor_parallel_args(args.baseline_tensor_parallel_size),
                        *common_vllm_args(args),
                    ],
                    args.pd_cpu_affinity,
                ),
                point_dir / "logs" / f"baseline_agg_gpu{group_slug}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + worker_idx,
                        **fpm_env(args, worker_idx),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )
    for worker_idx, group in enumerate(groups):
        group_slug = gpu_group_slug(group)
        wait_for(
            f"baseline worker GPU{group_slug}",
            lambda worker_idx=worker_idx: url_ready(
                f"http://127.0.0.1:{args.system_port_base + worker_idx}/health"
            ),
            procs,
            args.startup_timeout,
        )
    wait_for(
        "baseline frontend model list",
        lambda: url_ok(f"http://127.0.0.1:{args.http_port}/v1/models"),
        procs,
        args.startup_timeout,
    )
    return procs


def launch_epd(
    args: argparse.Namespace, point_dir: Path, env: dict[str, str]
) -> list[ManagedProcess]:
    encoder_gpus, prefill_gpus, decode_gpus = epd_worker_gpus(args)
    encoder_groups = gpu_groups(
        encoder_gpus, args.epd_encoder_tensor_parallel_size, "EPD encoder"
    )
    prefill_groups = gpu_groups(
        prefill_gpus, args.epd_pd_tensor_parallel_size, "EPD prefill"
    )
    decode_groups = gpu_groups(
        decode_gpus, args.epd_pd_tensor_parallel_size, "EPD decode"
    )

    procs = [launch_frontend(args, point_dir, env, "epd")]
    duplicate_encoder_groups = has_duplicate_gpu_groups(encoder_groups)
    for encoder_idx, group in enumerate(encoder_groups):
        group_slug = gpu_group_slug(group)
        log_name = (
            f"epd_encoder{encoder_idx}_gpu{group_slug}.log"
            if duplicate_encoder_groups
            else (
                "epd_encoder.log"
                if len(encoder_groups) == 1 and group == [0]
                else f"epd_encoder_gpu{group_slug}.log"
            )
        )
        procs.append(
            ManagedProcess(
                (
                    f"epd-encoder{encoder_idx}-gpu{group_slug}"
                    if duplicate_encoder_groups
                    else f"epd-encoder-gpu{group_slug}"
                ),
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--enable-multimodal",
                        "--disaggregation-mode",
                        "encode",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_e),
                        "--max-num-batched-tokens",
                        str(args.encoder_max_num_batched_tokens),
                        "--enforce-eager",
                        *kv_transfer_args(20080 + encoder_idx),
                        *tensor_parallel_args(args.epd_encoder_tensor_parallel_size),
                        *common_vllm_args(args),
                    ],
                    args.encoder_cpu_affinity,
                ),
                point_dir / "logs" / log_name,
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + encoder_idx,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20097 + encoder_idx,
                        **fpm_env(args, encoder_idx),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    for prefill_idx, group in enumerate(prefill_groups):
        port_offset = len(encoder_groups) + prefill_idx
        group_slug = gpu_group_slug(group)
        procs.append(
            ManagedProcess(
                f"epd-prefill-gpu{group_slug}",
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--enable-multimodal",
                        "--disaggregation-mode",
                        "prefill",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_pd),
                        *tensor_parallel_args(args.epd_pd_tensor_parallel_size),
                        *epd_pd_vllm_args(args),
                    ],
                    args.pd_cpu_affinity,
                ),
                point_dir / "logs" / f"epd_prefill_gpu{group_slug}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + port_offset,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20097 + port_offset,
                        **fpm_env(args, port_offset),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    decode_port_base = len(encoder_groups) + len(prefill_groups)
    for decode_idx, group in enumerate(decode_groups):
        port_offset = decode_port_base + decode_idx
        group_slug = gpu_group_slug(group)
        procs.append(
            ManagedProcess(
                f"epd-decode-gpu{group_slug}",
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--enable-multimodal",
                        "--disaggregation-mode",
                        "decode",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_pd),
                        *tensor_parallel_args(args.epd_pd_tensor_parallel_size),
                        *epd_pd_vllm_args(args),
                    ],
                    args.pd_cpu_affinity,
                ),
                point_dir / "logs" / f"epd_decode_gpu{group_slug}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + port_offset,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20097 + port_offset,
                        **fpm_env(args, port_offset),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    for encoder_idx, group in enumerate(encoder_groups):
        group_slug = gpu_group_slug(group)
        wait_for(
            f"epd encoder GPU{group_slug} instance {encoder_idx}",
            lambda encoder_idx=encoder_idx: endpoint_ready(
                f"http://127.0.0.1:{args.system_port_base + encoder_idx}/health",
                "generate",
            ),
            procs,
            args.startup_timeout,
        )
    for prefill_idx, group in enumerate(prefill_groups):
        port_offset = len(encoder_groups) + prefill_idx
        group_slug = gpu_group_slug(group)
        wait_for(
            f"epd prefill GPU{group_slug}",
            lambda port_offset=port_offset: url_ready(
                f"http://127.0.0.1:{args.system_port_base + port_offset}/health"
            ),
            procs,
            args.startup_timeout,
        )
    for decode_idx, group in enumerate(decode_groups):
        port_offset = decode_port_base + decode_idx
        group_slug = gpu_group_slug(group)
        wait_for(
            f"epd decode GPU{group_slug}",
            lambda port_offset=port_offset: url_ready(
                f"http://127.0.0.1:{args.system_port_base + port_offset}/health"
            ),
            procs,
            args.startup_timeout,
        )
    wait_for(
        "epd frontend model list",
        lambda: url_ok(f"http://127.0.0.1:{args.http_port}/v1/models"),
        procs,
        args.startup_timeout,
    )
    return procs

def launch_e_pd(
    args: argparse.Namespace, point_dir: Path, env: dict[str, str]
) -> list[ManagedProcess]:
    encoder_gpus, pd_gpus = e_pd_worker_gpus(args)
    encoder_groups = gpu_groups(
        encoder_gpus, args.epd_encoder_tensor_parallel_size, "E/PD encoder"
    )
    pd_groups = gpu_groups(pd_gpus, args.epd_pd_tensor_parallel_size, "E/PD PD")

    procs = [launch_frontend(args, point_dir, env, "e_pd")]
    duplicate_encoder_groups = has_duplicate_gpu_groups(encoder_groups)
    for encoder_idx, group in enumerate(encoder_groups):
        group_slug = gpu_group_slug(group)
        log_name = (
            f"e_pd_encoder{encoder_idx}_gpu{group_slug}.log"
            if duplicate_encoder_groups
            else f"e_pd_encoder_gpu{group_slug}.log"
        )
        procs.append(
            ManagedProcess(
                (
                    f"e-pd-encoder{encoder_idx}-gpu{group_slug}"
                    if duplicate_encoder_groups
                    else f"e-pd-encoder-gpu{group_slug}"
                ),
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--enable-multimodal",
                        "--disaggregation-mode",
                        "encode",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_e),
                        "--max-num-batched-tokens",
                        str(args.encoder_max_num_batched_tokens),
                        "--enforce-eager",
                        *kv_transfer_args(20080 + encoder_idx),
                        *tensor_parallel_args(args.epd_encoder_tensor_parallel_size),
                        *common_vllm_args(args),
                    ],
                    args.encoder_cpu_affinity,
                ),
                point_dir / "logs" / log_name,
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + encoder_idx,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20097 + encoder_idx,
                        **fpm_env(args, encoder_idx),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    for pd_idx, group in enumerate(pd_groups):
        port_offset = len(encoder_groups) + pd_idx
        group_slug = gpu_group_slug(group)
        procs.append(
            ManagedProcess(
                f"e-pd-pd-gpu{group_slug}",
                with_cpu_affinity(
                    [
                        sys.executable,
                        "-m",
                        "dynamo.vllm",
                        "--route-to-encoder",
                        "--enable-multimodal",
                        "--enable-mm-embeds",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization_pd),
                        *tensor_parallel_args(args.epd_pd_tensor_parallel_size),
                        *epd_pd_vllm_args(args),
                    ],
                    args.pd_cpu_affinity,
                ),
                point_dir / "logs" / f"e_pd_pd_gpu{group_slug}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": cuda_visible_devices(group),
                        "DYN_SYSTEM_PORT": args.system_port_base + port_offset,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20097 + port_offset,
                        **fpm_env(args, port_offset),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    for encoder_idx, group in enumerate(encoder_groups):
        group_slug = gpu_group_slug(group)
        wait_for(
            f"e_pd encoder GPU{group_slug} generate endpoint",
            lambda encoder_idx=encoder_idx: endpoint_ready(
                f"http://127.0.0.1:{args.system_port_base + encoder_idx}/health",
                "generate",
            ),
            procs,
            args.startup_timeout,
        )
    for pd_idx, group in enumerate(pd_groups):
        port_offset = len(encoder_groups) + pd_idx
        group_slug = gpu_group_slug(group)
        wait_for(
            f"e_pd PD GPU{group_slug}",
            lambda port_offset=port_offset: url_ready(
                f"http://127.0.0.1:{args.system_port_base + port_offset}/health"
            ),
            procs,
            args.startup_timeout,
        )
    wait_for(
        "e_pd frontend model list",
        lambda: url_ok(f"http://127.0.0.1:{args.http_port}/v1/models"),
        procs,
        args.startup_timeout,
    )
    return procs


def launch_hybrid_epd(
    args: argparse.Namespace, point_dir: Path, env: dict[str, str]
) -> list[ManagedProcess]:
    prefill_gpus, decode_gpus = hybrid_epd_worker_gpus(args)
    encode_roles = args.hybrid_epd_encode_roles
    hybrid_client_args = []
    if args.hybrid_epd_role_endpoints:
        hybrid_client_args = [
            "--hybrid-epd-encode-client-roles",
            args.hybrid_epd_encode_client_roles,
            "--hybrid-epd-encode-routing-policy",
            args.hybrid_epd_encode_routing_policy,
            "--hybrid-epd-prefill-inflight-limit",
            str(args.hybrid_epd_prefill_inflight_limit),
        ]
    if args.hybrid_epd_local_encode_inflight_limit is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-encode-inflight-limit",
                str(args.hybrid_epd_local_encode_inflight_limit),
            ]
        )
    if args.hybrid_epd_local_encode_head_batches is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-encode-head-batches",
                str(args.hybrid_epd_local_encode_head_batches),
            ]
        )
    if args.hybrid_epd_local_encode_max_active_decode is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-encode-max-active-decode",
                str(args.hybrid_epd_local_encode_max_active_decode),
            ]
        )
    if args.hybrid_epd_local_encode_decode_pressure_head_batches is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-encode-decode-pressure-head-batches",
                str(args.hybrid_epd_local_encode_decode_pressure_head_batches),
                "--hybrid-epd-local-encode-decode-pressure-threshold",
                str(args.hybrid_epd_local_encode_decode_pressure_threshold),
            ]
        )
    if args.hybrid_epd_local_encode_max_active_prefill is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-encode-max-active-prefill",
                str(args.hybrid_epd_local_encode_max_active_prefill),
            ]
        )
    if args.hybrid_epd_encode_inflight_limit is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-encode-inflight-limit",
                str(args.hybrid_epd_encode_inflight_limit),
            ]
        )
    if args.hybrid_epd_local_prefill_inflight_limit is not None:
        hybrid_client_args.extend(
            [
                "--hybrid-epd-local-prefill-inflight-limit",
                str(args.hybrid_epd_local_prefill_inflight_limit),
            ]
        )
    decode_aux_prefill_limit_args = (
        [
            "--hybrid-epd-aux-prefill-inflight-limit",
            str(args.hybrid_epd_aux_prefill_inflight_limit),
        ]
        if args.hybrid_epd_aux_prefill_inflight_limit is not None
        else []
    )
    hybrid_all_capability_args = (
        ["--hybrid-epd-all-capabilities"] if args.hybrid_epd_all_capabilities else []
    )
    hybrid_aux_prefill_args = (
        []
        if args.hybrid_epd_serve_aux_prefill
        else ["--no-hybrid-epd-serve-aux-prefill"]
    )
    hybrid_aux_decode_args = (
        [] if args.hybrid_epd_serve_aux_decode else ["--no-hybrid-epd-serve-aux-decode"]
    )

    procs = [launch_frontend(args, point_dir, env, "hybrid_epd")]
    for prefill_idx, gpu in enumerate(prefill_gpus):
        prefill_encode_args = (
            []
            if encode_roles in {"all", "prefill"}
            else ["--no-hybrid-epd-serve-encode"]
        )
        prefill_role_args = (
            ["--hybrid-epd-encode-endpoint-role", "prefill"]
            if args.hybrid_epd_role_endpoints
            else []
        )
        procs.append(
            ManagedProcess(
                f"hybrid-epd-prefill-gpu{gpu}",
                [
                    sys.executable,
                    "-m",
                    "dynamo.vllm",
                    "--route-to-encoder",
                    "--enable-multimodal",
                    "--disaggregation-mode",
                    "prefill",
                    "--enable-mm-embeds",
                    "--hybrid-epd-worker",
                    *hybrid_all_capability_args,
                    *hybrid_aux_decode_args,
                    *prefill_encode_args,
                    *prefill_role_args,
                    *hybrid_client_args,
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization_pd),
                    *kv_transfer_args(20180 + prefill_idx),
                    *epd_pd_vllm_args(args),
                ],
                point_dir / "logs" / f"hybrid_epd_prefill_gpu{gpu}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "DYN_SYSTEM_PORT": args.system_port_base + prefill_idx,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20280 + prefill_idx,
                        **fpm_env(args, prefill_idx),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    decode_port_base = len(prefill_gpus)
    for decode_idx, gpu in enumerate(decode_gpus):
        port_offset = decode_port_base + decode_idx
        decode_encode_args = (
            []
            if encode_roles in {"all", "decode"}
            else ["--no-hybrid-epd-serve-encode"]
        )
        decode_role_args = (
            ["--hybrid-epd-encode-endpoint-role", "decode"]
            if args.hybrid_epd_role_endpoints
            else []
        )
        direct_text_args = (
            [
                "--hybrid-epd-direct-text-model-name",
                args.hybrid_epd_direct_text_model_name,
            ]
            if args.hybrid_epd_direct_text_model_name
            else []
        )
        procs.append(
            ManagedProcess(
                f"hybrid-epd-decode-gpu{gpu}",
                [
                    sys.executable,
                    "-m",
                    "dynamo.vllm",
                    "--enable-multimodal",
                    "--disaggregation-mode",
                    "decode",
                    "--enable-mm-embeds",
                    "--hybrid-epd-worker",
                    *hybrid_all_capability_args,
                    *hybrid_aux_prefill_args,
                    *decode_aux_prefill_limit_args,
                    *decode_encode_args,
                    *decode_role_args,
                    *hybrid_client_args,
                    *direct_text_args,
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization_pd),
                    *kv_transfer_args(20190 + decode_idx),
                    *epd_pd_vllm_args(args),
                ],
                point_dir / "logs" / f"hybrid_epd_decode_gpu{gpu}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "DYN_SYSTEM_PORT": args.system_port_base + port_offset,
                        "VLLM_NIXL_SIDE_CHANNEL_PORT": 20290 + decode_idx,
                        **fpm_env(args, port_offset),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )

    for prefill_idx, gpu in enumerate(prefill_gpus):
        port = args.system_port_base + prefill_idx
        wait_for(
            f"hybrid EPD prefill GPU{gpu} generate endpoint",
            lambda port=port: endpoint_ready(
                f"http://127.0.0.1:{port}/health", "generate"
            ),
            procs,
            args.startup_timeout,
        )
        if encode_roles in {"all", "prefill"}:
            wait_for(
                f"hybrid EPD prefill GPU{gpu} encode endpoint",
                lambda port=port: endpoint_ready(
                    f"http://127.0.0.1:{port}/health", "embed"
                ),
                procs,
                args.startup_timeout,
            )
    for decode_idx, gpu in enumerate(decode_gpus):
        port = args.system_port_base + decode_port_base + decode_idx
        wait_for(
            f"hybrid EPD decode GPU{gpu} generate endpoint",
            lambda port=port: endpoint_ready(
                f"http://127.0.0.1:{port}/health", "generate"
            ),
            procs,
            args.startup_timeout,
        )
        if encode_roles in {"all", "decode"}:
            wait_for(
                f"hybrid EPD decode GPU{gpu} encode endpoint",
                lambda port=port: endpoint_ready(
                    f"http://127.0.0.1:{port}/health", "embed"
                ),
                procs,
                args.startup_timeout,
            )
    wait_for(
        "hybrid EPD frontend model list",
        lambda: url_ok(f"http://127.0.0.1:{args.http_port}/v1/models"),
        procs,
        args.startup_timeout,
    )
    return procs


class RunningBench:
    def __init__(
        self, name: str, cmd: list[str], log_path: Path, env: dict[str, str]
    ) -> None:
        self.name = name
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(VLLM_SRC),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait(self) -> None:
        code = self.proc.wait()
        self.log_file.close()
        if code != 0:
            raise RuntimeError(
                f"{self.name} failed with code {code}; see {self.log_path}"
            )


def client_model_for_stream(
    args: argparse.Namespace, topology: str, stream: str | None
) -> str:
    if (
        topology == "hybrid_epd"
        and stream == "text"
        and args.hybrid_epd_direct_text_model_name
    ):
        return args.hybrid_epd_direct_text_model_name
    return args.model


def replay_client_cmd(
    args: argparse.Namespace,
    topology: str,
    result_dir: Path,
    stream: str | None,
) -> tuple[list[str], str]:
    suffix = f"_{stream}" if stream else ""
    result_name = f"{topology}_replay_{args.replay_workload}{suffix}.json"
    client_model = client_model_for_stream(args, topology, stream)
    cmd = [
        sys.executable,
        str(FIXED_CLIENT),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.http_port),
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        client_model,
        "--tokenizer-model",
        args.model,
        "--replay-jsonl",
        str(args.replay_jsonl),
        "--replay-workload",
        args.replay_workload,
        "--settle-seconds",
        str(args.settle_seconds),
        "--max-inflight",
        str(args.max_inflight),
        "--num-warmups",
        str(args.num_warmups),
        "--slo-ttft-ms",
        str(args.slo_ttft_ms),
        "--slo-tpot-ms",
        str(args.slo_tpot_ms),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_name,
        "--metadata",
        f"topology={topology}",
        f"replay_workload={args.replay_workload}",
        f"client_model={client_model}",
        "stack=dynamo_vllm",
    ]
    if stream:
        cmd += ["--replay-stream", stream]
    if (
        topology == "hybrid_epd"
        and args.hybrid_epd_direct_text_bypass
        and stream == "text"
    ):
        cmd += [
            "--extra-body-json",
            json.dumps(
                {"nvext": {"cache_salt": "dynamo_hybrid_epd_text_only_direct_decode"}},
                separators=(",", ":"),
            ),
        ]
    return cmd, result_name


def run_replay_clients(
    args: argparse.Namespace,
    point_dir: Path,
    topology: str,
    env: dict[str, str],
) -> None:
    run_prewarm_clients(args, point_dir, topology, env)
    result_dir = args.run_dir / "results" / topology / "replay"
    result_dir.mkdir(parents=True, exist_ok=True)
    streams = split_csv(args.replay_client_streams)
    selected_streams: list[str | None] = streams or [None]
    benches: list[RunningBench] = []
    for stream in selected_streams:
        cmd, _ = replay_client_cmd(args, topology, result_dir, stream)
        suffix = f"_{stream}" if stream else ""
        benches.append(
            RunningBench(
                f"{topology}-replay{suffix}",
                cmd,
                point_dir
                / "logs"
                / f"bench_{topology}_replay_{args.replay_workload}{suffix}.log",
                env,
            )
        )
    for bench in benches:
        bench.wait()


def run_prewarm_clients(
    args: argparse.Namespace,
    point_dir: Path,
    topology: str,
    env: dict[str, str],
) -> None:
    if args.prewarm_requests_per_stream <= 0:
        return
    result_dir = args.run_dir / "results" / topology / "prewarm"
    result_dir.mkdir(parents=True, exist_ok=True)
    streams = split_csv(args.replay_client_streams)
    selected_streams: list[str | None] = streams or [None]
    for stream in selected_streams:
        cmd, _ = replay_client_cmd(args, topology, result_dir, stream)
        suffix = f"_{stream}" if stream else ""
        cmd += [
            "--warmup-only",
            "--num-warmups",
            str(args.prewarm_requests_per_stream),
            "--result-filename",
            f"{topology}_prewarm_{args.replay_workload}{suffix}.json",
            "--request-id-prefix",
            f"{topology}-prewarm{suffix}-",
        ]
        prewarm = RunningBench(
            f"{topology}-prewarm{suffix}",
            cmd,
            point_dir
            / "logs"
            / f"prewarm_{topology}_{args.replay_workload}{suffix}.log",
            env,
        )
        prewarm.wait()
    if args.prewarm_after_sleep_s > 0:
        time.sleep(args.prewarm_after_sleep_s)


def point_dir(args: argparse.Namespace, topology: str) -> Path:
    return args.run_dir / "points" / f"{topology}_replay_{args.replay_workload}"


def collect_manifest(args: argparse.Namespace) -> None:
    workload = read_replay_workload(args.replay_jsonl, args.replay_workload)
    topologies = split_csv(args.topologies)
    baseline_groups = baseline_gpu_groups(args)
    if "e_pd" in topologies:
        e_pd_encoder_gpus, e_pd_pd_gpus = e_pd_worker_gpus(args)
        e_pd_encoder_groups = gpu_groups(
            e_pd_encoder_gpus,
            args.epd_encoder_tensor_parallel_size,
            "E/PD encoder",
        )
        e_pd_pd_groups = gpu_groups(
            e_pd_pd_gpus, args.epd_pd_tensor_parallel_size, "E/PD PD"
        )
    else:
        e_pd_encoder_gpus, e_pd_pd_gpus = [], []
        e_pd_encoder_groups, e_pd_pd_groups = [], []
    if "epd" in topologies:
        encoder_gpus, prefill_gpus, decode_gpus = epd_worker_gpus(args)
        encoder_groups = gpu_groups(
            encoder_gpus, args.epd_encoder_tensor_parallel_size, "EPD encoder"
        )
        prefill_groups = gpu_groups(
            prefill_gpus, args.epd_pd_tensor_parallel_size, "EPD prefill"
        )
        decode_groups = gpu_groups(
            decode_gpus, args.epd_pd_tensor_parallel_size, "EPD decode"
        )
    else:
        encoder_gpus, prefill_gpus, decode_gpus = [], [], []
        encoder_groups, prefill_groups, decode_groups = [], [], []
    hybrid_prefill_gpus, hybrid_decode_gpus = hybrid_epd_worker_gpus(args)
    probes: dict[str, Any] = {}
    probe_cmds = {
        "dynamo_git": ["git", "-C", str(DYNAMO_ROOT), "rev-parse", "HEAD"],
        "dynamo_runtime_git": [
            "git",
            "-C",
            str(DYNAMO_RUNTIME_ROOT),
            "rev-parse",
            "HEAD",
        ],
        "vllm_git": ["git", "-C", str(VLLM_SRC), "rev-parse", "HEAD"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    }
    for key, cmd in probe_cmds.items():
        try:
            probes[key] = subprocess.check_output(
                cmd, text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception as exc:  # noqa: BLE001
            probes[key] = f"ERROR: {exc}"
    write_json(
        args.run_dir / "scenario_manifest.json",
        {
            "created_utc": now_slug(),
            "scenario": "dynamo_vllm_text_image_fixed_replay",
            "model": args.model,
            "replay_jsonl": str(args.replay_jsonl),
            "replay_workload": args.replay_workload,
            "replay_client_streams": split_csv(args.replay_client_streams),
            "workload": workload,
            "topologies": topologies,
            "server": {
                "baseline": (
                    "Dynamo frontend + "
                    f"{len(baseline_groups)} vLLM aggregated worker(s), "
                    f"TP={args.baseline_tensor_parallel_size}, "
                    f"GPU groups={baseline_groups}"
                ),
                "hybrid": (
                    "Dynamo frontend + four single-GPU vLLM aggregated workers "
                    "that also serve the shared encode endpoint"
                ),
                "hybrid_epd": (
                    f"Dynamo frontend + {len(hybrid_prefill_gpus)} prefill worker(s) + "
                    f"{len(hybrid_decode_gpus)} decode worker(s); every P/D worker "
                    "can serve encode, filtered by "
                    f"hybrid_epd_encode_roles={args.hybrid_epd_encode_roles}; "
                    f"role_endpoints={args.hybrid_epd_role_endpoints}; "
                    f"client_roles={args.hybrid_epd_encode_client_roles}; "
                    f"policy={args.hybrid_epd_encode_routing_policy}; "
                    "local_encode_inflight_limit="
                    f"{args.hybrid_epd_local_encode_inflight_limit}; "
                    "local_encode_max_active_decode="
                    f"{args.hybrid_epd_local_encode_max_active_decode}; "
                    "local_encode_max_active_prefill="
                    f"{args.hybrid_epd_local_encode_max_active_prefill}; "
                    "encode_inflight_limit="
                    f"{args.hybrid_epd_encode_inflight_limit}; "
                    "local_prefill_inflight_limit="
                    f"{args.hybrid_epd_local_prefill_inflight_limit}; "
                    "prefill_worker_limit="
                    f"{args.hybrid_epd_prefill_worker_limit}; "
                    "prefill_worker_burst_limit="
                    f"{args.hybrid_epd_prefill_worker_burst_limit}; "
                    "prefill_worker_burst_percent="
                    f"{args.hybrid_epd_prefill_worker_burst_percent}; "
                    "prefill_worker_backlog_admission="
                    f"{args.hybrid_epd_prefill_worker_backlog_admission}; "
                    "prefill_worker_backlog_high_watermark_percent="
                    f"{args.hybrid_epd_prefill_worker_backlog_high_watermark_percent}; "
                    "mm_decode_backlog_guard_max_requests="
                    f"{args.hybrid_epd_mm_decode_backlog_guard_max_requests}; "
                    "decode_block_pressure_guard_percent="
                    f"{args.hybrid_epd_decode_block_pressure_guard_percent}; "
                    "decode_block_pressure_guard_min_available_workers="
                    f"{args.hybrid_epd_decode_block_pressure_guard_min_available_workers}; "
                    "decode_request_load_guard_max="
                    f"{args.hybrid_epd_decode_request_load_guard_max}; "
                    "decode_request_load_guard_min_available_workers="
                    f"{args.hybrid_epd_decode_request_load_guard_min_available_workers}; "
                    "decode_request_load_guard_prefill_worker_limit="
                    f"{args.hybrid_epd_decode_request_load_guard_prefill_worker_limit}; "
                    "aux_prefill_decode_request_load_max="
                    f"{args.hybrid_epd_aux_prefill_decode_request_load_max}; "
                    "aux_prefill_include_unknown_decode_load="
                    f"{args.hybrid_epd_aux_prefill_include_unknown_decode_load}; "
                    "reserved_text_decode_workers="
                    f"{args.hybrid_epd_reserved_text_decode_workers}; "
                    "exclude_reserved_text_decode_from_mm_decode="
                    f"{args.hybrid_epd_exclude_reserved_text_decode_from_mm_decode}; "
                    f"all_capabilities={args.hybrid_epd_all_capabilities}; "
                    f"serve_aux_prefill={args.hybrid_epd_serve_aux_prefill}; "
                    f"serve_aux_decode={args.hybrid_epd_serve_aux_decode}"
                ),
                "epd": (
                    f"Dynamo frontend + {len(encoder_groups)} encode worker(s) + "
                    f"{len(prefill_groups)} prefill worker(s) + "
                    f"{len(decode_groups)} decode worker(s); "
                    f"E TP={args.epd_encoder_tensor_parallel_size}, "
                    f"PD TP={args.epd_pd_tensor_parallel_size}"
                ),
                "e_pd": (
                    f"Dynamo frontend + {len(e_pd_encoder_groups)} encode worker(s) + "
                    f"{len(e_pd_pd_groups)} aggregated PD worker(s); "
                    f"E TP={args.epd_encoder_tensor_parallel_size}, "
                    f"PD TP={args.epd_pd_tensor_parallel_size}"
                ),
                "baseline_gpu_groups": baseline_groups,
                "baseline_tensor_parallel_size": args.baseline_tensor_parallel_size,
                "e_pd_encoder_gpus": e_pd_encoder_gpus,
                "e_pd_pd_gpus": e_pd_pd_gpus,
                "e_pd_encoder_gpu_groups": e_pd_encoder_groups,
                "e_pd_pd_gpu_groups": e_pd_pd_groups,
                "epd_encoder_gpus": encoder_gpus,
                "epd_prefill_gpus": prefill_gpus,
                "epd_decode_gpus": decode_gpus,
                "epd_encoder_gpu_groups": encoder_groups,
                "epd_prefill_gpu_groups": prefill_groups,
                "epd_decode_gpu_groups": decode_groups,
                "epd_encoder_tensor_parallel_size": args.epd_encoder_tensor_parallel_size,
                "epd_pd_tensor_parallel_size": args.epd_pd_tensor_parallel_size,
                "hybrid_epd_prefill_gpus": hybrid_prefill_gpus,
                "hybrid_epd_decode_gpus": hybrid_decode_gpus,
                "hybrid_epd_encode_roles": args.hybrid_epd_encode_roles,
                "hybrid_epd_role_endpoints": args.hybrid_epd_role_endpoints,
                "hybrid_epd_encode_client_roles": args.hybrid_epd_encode_client_roles,
                "hybrid_epd_encode_routing_policy": args.hybrid_epd_encode_routing_policy,
                "hybrid_epd_prefill_inflight_limit": args.hybrid_epd_prefill_inflight_limit,
                "hybrid_epd_local_encode_inflight_limit": args.hybrid_epd_local_encode_inflight_limit,
                "hybrid_epd_local_encode_max_active_decode": args.hybrid_epd_local_encode_max_active_decode,
                "hybrid_epd_local_encode_max_active_prefill": args.hybrid_epd_local_encode_max_active_prefill,
                "hybrid_epd_encode_inflight_limit": args.hybrid_epd_encode_inflight_limit,
                "hybrid_epd_local_prefill_inflight_limit": args.hybrid_epd_local_prefill_inflight_limit,
                "hybrid_epd_prefill_worker_limit": args.hybrid_epd_prefill_worker_limit,
                "hybrid_epd_prefill_worker_burst_limit": args.hybrid_epd_prefill_worker_burst_limit,
                "hybrid_epd_prefill_worker_burst_percent": args.hybrid_epd_prefill_worker_burst_percent,
                "hybrid_epd_prefill_worker_backlog_admission": args.hybrid_epd_prefill_worker_backlog_admission,
                "hybrid_epd_prefill_worker_backlog_high_watermark_percent": args.hybrid_epd_prefill_worker_backlog_high_watermark_percent,
                "hybrid_epd_mm_decode_backlog_guard_max_requests": args.hybrid_epd_mm_decode_backlog_guard_max_requests,
                "hybrid_epd_decode_block_pressure_guard_percent": args.hybrid_epd_decode_block_pressure_guard_percent,
                "hybrid_epd_decode_block_pressure_guard_min_available_workers": args.hybrid_epd_decode_block_pressure_guard_min_available_workers,
                "hybrid_epd_decode_request_load_guard_max": args.hybrid_epd_decode_request_load_guard_max,
                "hybrid_epd_decode_request_load_guard_min_available_workers": args.hybrid_epd_decode_request_load_guard_min_available_workers,
                "hybrid_epd_decode_request_load_guard_prefill_worker_limit": args.hybrid_epd_decode_request_load_guard_prefill_worker_limit,
                "hybrid_epd_aux_prefill_decode_request_load_max": args.hybrid_epd_aux_prefill_decode_request_load_max,
                "hybrid_epd_aux_prefill_include_unknown_decode_load": args.hybrid_epd_aux_prefill_include_unknown_decode_load,
                "hybrid_epd_reserved_text_decode_workers": args.hybrid_epd_reserved_text_decode_workers,
                "hybrid_epd_exclude_reserved_text_decode_from_mm_decode": args.hybrid_epd_exclude_reserved_text_decode_from_mm_decode,
                "hybrid_epd_direct_text_model_name": args.hybrid_epd_direct_text_model_name,
                "hybrid_epd_direct_text_bypass": args.hybrid_epd_direct_text_bypass,
                "hybrid_epd_all_capabilities": args.hybrid_epd_all_capabilities,
                "hybrid_epd_serve_aux_prefill": args.hybrid_epd_serve_aux_prefill,
                "hybrid_epd_serve_aux_decode": args.hybrid_epd_serve_aux_decode,
                "frontend_router_mode": args.frontend_router_mode,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "images_per_request": args.images_per_request,
                "videos_per_request": args.videos_per_request,
                "disable_prefix_caching": args.disable_prefix_caching,
                "cuda_graph": "enabled on agg/PD workers by not passing --enforce-eager",
                "encoder_enforce_eager": True,
                "gpu_memory_utilization_e": args.gpu_memory_utilization_e,
                "gpu_memory_utilization_pd": args.gpu_memory_utilization_pd,
                "encoder_max_num_batched_tokens": args.encoder_max_num_batched_tokens,
                "encoder_max_concurrent": args.encoder_max_concurrent,
                "encode_batch_size": args.encode_batch_size,
                "encode_dispatch_concurrency": args.encode_dispatch_concurrency,
                "eworker_microbatch_max_items": args.eworker_microbatch_max_items,
                "eworker_microbatch_wait_ms": args.eworker_microbatch_wait_ms,
                "eworker_parallel_preprocess": args.eworker_parallel_preprocess,
                "eworker_inline_encode": args.eworker_inline_encode,
                "eworker_microbatch_thread": args.eworker_microbatch_thread,
                "eworker_torch_num_threads": args.eworker_torch_num_threads,
                "eworker_torch_num_interop_threads": args.eworker_torch_num_interop_threads,
                "eworker_timing_breakdown": args.eworker_timing_breakdown,
                "eworker_executor_workers": args.eworker_executor_workers,
                "eworker_image_decode_executor_workers": args.eworker_image_decode_executor_workers,
                "mm_torch_num_threads": args.mm_torch_num_threads,
                "mm_executor_workers": args.mm_executor_workers,
                "thread_env_overrides": thread_env_overrides(),
                "frontend_cpu_affinity": args.frontend_cpu_affinity,
                "encoder_cpu_affinity": args.encoder_cpu_affinity,
                "pd_cpu_affinity": args.pd_cpu_affinity,
                "mm_processor_cache_gb": args.mm_processor_cache_gb,
                "dynamo_embedding_cache_capacity_gb": args.dynamo_embedding_cache_capacity_gb,
                "enable_encoder_cache": args.enable_encoder_cache,
                "embedding_transfer_mode": args.embedding_transfer_mode,
                "ucx_tls": args.ucx_tls,
                "ucx_net_devices": args.ucx_net_devices,
                "nixl_write_receiver_buffer_gb": args.nixl_write_receiver_buffer_gb,
                "nixl_write_receive_timeout_seconds": args.nixl_write_receive_timeout_seconds,
                "nixl_write_receiver_device": args.nixl_write_receiver_device,
                "forwardpass_metric_port_base": args.forwardpass_metric_port_base,
            },
            "client": {
                "settle_seconds": args.settle_seconds,
                "max_inflight": args.max_inflight,
                "num_warmups": args.num_warmups,
                "prewarm_requests_per_stream": args.prewarm_requests_per_stream,
                "prewarm_after_sleep_s": args.prewarm_after_sleep_s,
                "slo_ttft_ms": args.slo_ttft_ms,
                "slo_tpot_ms": args.slo_tpot_ms,
            },
            **probes,
        },
    )


def launch_hybrid(
    args: argparse.Namespace, point_dir: Path, env: dict[str, str]
) -> list[ManagedProcess]:
    procs = [launch_frontend(args, point_dir, env, "hybrid")]
    for gpu in range(4):
        procs.append(
            ManagedProcess(
                f"hybrid-agg-encode-gpu{gpu}",
                [
                    sys.executable,
                    "-m",
                    "dynamo.vllm",
                    "--enable-multimodal",
                    "--disaggregation-mode",
                    "agg",
                    "--hybrid-epd-worker",
                    "--enable-mm-embeds",
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization_pd),
                    *epd_pd_vllm_args(args),
                ],
                point_dir / "logs" / f"hybrid_agg_encode_gpu{gpu}.log",
                env=with_env(
                    env,
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "DYN_SYSTEM_PORT": args.system_port_base + gpu,
                        **fpm_env(args, gpu),
                    },
                ),
                cwd=DYNAMO_ROOT / "examples" / "backends" / "vllm",
            )
        )
    for gpu in range(4):
        wait_for(
            f"hybrid worker GPU{gpu}",
            lambda gpu=gpu: url_ready(
                f"http://127.0.0.1:{args.system_port_base + gpu}/health"
            ),
            procs,
            args.startup_timeout,
        )
        wait_for(
            f"hybrid worker GPU{gpu} generate endpoint",
            lambda gpu=gpu: endpoint_ready(
                f"http://127.0.0.1:{args.system_port_base + gpu}/health",
                "generate",
            ),
            procs,
            args.startup_timeout,
        )
    wait_for(
        "hybrid frontend model list",
        lambda: url_ok(f"http://127.0.0.1:{args.http_port}/v1/models"),
        procs,
        args.startup_timeout,
    )
    return procs


def run_topology(args: argparse.Namespace, topology: str) -> None:
    point = point_dir(args, topology)
    point.mkdir(parents=True, exist_ok=True)
    namespace = f"dynamo-vllm-{topology}-{args.run_dir.name}".replace("_", "-")
    env = base_env(args, point, namespace)
    procs: list[ManagedProcess] = []
    sampler: ManagedProcess | None = None
    try:
        print(f"[{topology}] run_dir={point}", flush=True)
        etcd = launch_etcd(args, point)
        procs.append(etcd)
        wait_for(
            "etcd",
            lambda: url_ok(f"http://127.0.0.1:{args.etcd_client_port}/health"),
            procs,
            args.startup_timeout,
        )
        sampler = launch_gpu_sampler(point, topology)
        if topology == "baseline":
            procs.extend(launch_baseline(args, point, env))
        elif topology == "hybrid":
            procs.extend(launch_hybrid(args, point, env))
        elif topology == "hybrid_epd":
            procs.extend(launch_hybrid_epd(args, point, env))
        elif topology == "e_pd":
            procs.extend(launch_e_pd(args, point, env))
        elif topology == "epd":
            procs.extend(launch_epd(args, point, env))
        else:
            raise ValueError(f"Unknown topology {topology!r}")
        run_replay_clients(args, point, topology, env)
    finally:
        if sampler is not None:
            sampler.stop()
        for proc in reversed(procs):
            proc.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--topologies", default="baseline,epd")
    parser.add_argument("--replay-jsonl", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--replay-workload", default="text_image_qps32_1to1")
    parser.add_argument("--replay-client-streams", default="text,image")
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--system-port-base", type=int, default=18181)
    parser.add_argument("--etcd-client-port", type=int, default=2379)
    parser.add_argument("--etcd-peer-port", type=int, default=2380)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--frontend-router-mode", default="round-robin")
    parser.add_argument(
        "--frontend-active-decode-blocks-threshold",
        default=None,
        help=(
            "Forwarded to dynamo.frontend --active-decode-blocks-threshold. "
            "Use a float fraction or literal None as accepted by the frontend."
        ),
    )
    parser.add_argument(
        "--frontend-active-prefill-tokens-threshold",
        default=None,
        help=(
            "Forwarded to dynamo.frontend --active-prefill-tokens-threshold. "
            "Use this to mark workers busy when vLLM publishes large current "
            "prefill-token iterations."
        ),
    )
    parser.add_argument(
        "--frontend-active-prefill-tokens-threshold-frac",
        default=None,
        help=(
            "Forwarded to dynamo.frontend " "--active-prefill-tokens-threshold-frac."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-overload-spillover",
        action="store_true",
        help=(
            "When an admitted hybrid prefill worker set is fully overloaded, "
            "route to another free prefill worker instead of keeping the "
            "admitted set and risking runtime 503 backpressure."
        ),
    )
    parser.add_argument(
        "--runtime-allow-overloaded-dispatch",
        action="store_true",
        help=(
            "Keep overload state for routing hints, but do not hard-reject a "
            "direct/pinned dispatch when the selected worker is currently "
            "marked overloaded."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--images-per-request", type=int, default=4)
    parser.add_argument("--videos-per-request", type=int, default=0)
    parser.add_argument(
        "--baseline-tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for aggregated baseline workers. The four "
            "local GPUs are partitioned into consecutive TP groups, so TP=2 "
            "launches two agg workers on GPUs 0,1 and 2,3."
        ),
    )
    parser.add_argument("--epd-encoder-gpus", default="0")
    parser.add_argument(
        "--epd-encoder-gpu-groups",
        default=None,
        help=(
            "Explicit E worker GPU groups separated by ';'. Use this when "
            "multiple E workers should share the same physical GPU, e.g. "
            "'0;0' for two TP=1 E workers on GPU 0 or '0,1;0,1' for two "
            "TP=2 E workers on GPUs 0 and 1. Overrides --epd-encoder-gpus."
        ),
    )
    parser.add_argument("--epd-pd-gpus", default="1,2,3")
    parser.add_argument(
        "--allow-epd-gpu-overlap",
        action="store_true",
        help=(
            "Allow encoder and PD/prefill/decode workers to share physical GPUs. "
            "This is intended for colocated E/PD experiments."
        ),
    )
    parser.add_argument(
        "--epd-encoder-tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for each E/PD encoder worker. "
            "--epd-encoder-gpus is partitioned into consecutive TP groups; "
            "the number of groups is encoder-worker DP."
        ),
    )
    parser.add_argument(
        "--epd-pd-tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for each E/PD prefill/decode worker. "
            "Prefill and decode GPU lists are each partitioned into "
            "consecutive TP groups."
        ),
    )
    parser.add_argument(
        "--epd-prefill-gpus",
        default=None,
        help=(
            "Prefill GPUs for E/P/D mode. If omitted, the first GPU from "
            "--epd-pd-gpus is used for backward compatibility."
        ),
    )
    parser.add_argument(
        "--epd-decode-gpus",
        default=None,
        help=(
            "Decode GPUs for E/P/D mode. If omitted, all but the first GPU from "
            "--epd-pd-gpus are used for backward compatibility."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-encode-roles",
        default="all",
        choices=["all", "prefill", "decode"],
        help=(
            "Which hybrid EPD roles register into the shared encode endpoint. "
            "Use this for role-filtered encode routing experiments."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-role-endpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Register hybrid encode service under role-specific endpoints "
            "(encode_prefill.embed / encode_decode.embed) instead of the shared "
            "encode.embed pool."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-encode-client-roles",
        default="shared",
        help=(
            "Comma-separated encode endpoint roles for hybrid prefill dispatch. "
            "Use prefill,decode with --hybrid-epd-role-endpoints."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-encode-routing-policy",
        default="round-robin-roles",
        choices=[
            "round-robin-roles",
            "instance-weighted-round-robin",
            "least-inflight",
            "prefill-first",
        ],
    )
    parser.add_argument(
        "--hybrid-epd-prefill-inflight-limit",
        type=float,
        default=1.0,
        help=(
            "For prefill-first role-aware encode routing, use prefill encode "
            "while prefill in-flight encode requests per prefill instance are "
            "below this value, then spill to other roles."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-inflight-limit",
        type=float,
        default=None,
        help=(
            "Prefer a collocated hybrid encode endpoint only while this many "
            "local encode requests are in flight. Above the limit, spill to "
            "the configured role-aware encode policy. Omit for unbounded local "
            "preference; use 0 to always spill when another role is available."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-head-batches",
        type=int,
        default=None,
        help=(
            "Request-aware local encode policy: only the first N encode batches "
            "of each multimodal request may use the collocated encode endpoint; "
            "later batches spill to the configured role-aware policy."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-max-active-decode",
        type=int,
        default=None,
        help=(
            "Allow collocated hybrid encode only while the local worker's active "
            "decode request count is at or below this value. Use 0 to spill local "
            "encode whenever the worker is decoding."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-decode-pressure-head-batches",
        type=int,
        default=None,
        help=(
            "When the local worker's active decode count exceeds "
            "--hybrid-epd-local-encode-decode-pressure-threshold, override "
            "the normal local encode head-batch allowance with this value."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-decode-pressure-threshold",
        type=int,
        default=0,
        help=(
            "Active decode threshold for "
            "--hybrid-epd-local-encode-decode-pressure-head-batches."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-encode-max-active-prefill",
        type=int,
        default=None,
        help=(
            "Allow collocated hybrid encode only while the local worker's active "
            "prefill request count is at or below this value."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-encode-inflight-limit",
        type=int,
        default=None,
        help=(
            "Cap total in-flight multimodal encode batches issued by each "
            "hybrid worker. This provides backpressure before embedding handoff "
            "overwhelms the receiver buffer."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-local-prefill-inflight-limit",
        type=int,
        default=None,
        help=(
            "Cap concurrent prefill-role generate calls inside each hybrid "
            "all-capability worker. Requests above the cap wait before entering "
            "the local prefill handler."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-aux-prefill-inflight-limit",
        type=int,
        default=None,
        help=(
            "Cap concurrent aux-prefill generate calls on hybrid all-capability "
            "decode workers only, without throttling primary prefill workers."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-limit",
        type=int,
        default=None,
        help=(
            "Experimental all-capabilities admission control: limit multimodal "
            "requests that require synchronous prefill to the first N currently "
            "registered prefill endpoint workers. This reserves the remaining "
            "workers for decode after embedding/KV handoff."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-burst-limit",
        type=int,
        default=None,
        help=(
            "Experimental fractional admission control: for a stable hash-based "
            "fraction of multimodal prefill requests, use this worker limit "
            "instead of --hybrid-epd-prefill-worker-limit."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-burst-percent",
        type=int,
        default=None,
        help=(
            "Stable hash-based percentage of multimodal prefill requests that "
            "use --hybrid-epd-prefill-worker-burst-limit. Use with a base "
            "--hybrid-epd-prefill-worker-limit, e.g. base=2, burst=4, percent=50."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-backlog-max-tokens",
        type=int,
        default=None,
        help=(
            "Experimental router-local multimodal prefill backlog cap in prompt "
            "tokens. Requests wait before prefill dispatch instead of marking "
            "workers runtime-overloaded."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-backlog-max-requests",
        type=int,
        default=None,
        help=(
            "Experimental router-local cap on concurrent multimodal prefill "
            "requests. Applies only to requests that require embedding/KV handoff."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-reserved-text-decode-workers",
        type=int,
        default=None,
        help=(
            "Experimental hybrid E/P/D partition: reserve the highest-N decode "
            "workers for text-only direct decode. Multimodal prefill admission "
            "avoids these workers where possible."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-exclude-reserved-text-decode-from-mm-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When text decode workers are reserved, also remove them from "
            "multimodal decode routing if another decode worker remains."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-admission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Adjust the effective multimodal prefill burst percentage from the "
            "recent text/image arrival mix. This keeps hash-stable admission but "
            "changes the threshold online without restarting workers."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-window",
        type=int,
        default=None,
        help=(
            "Number of recent router arrivals used to estimate multimodal share "
            "for dynamic prefill admission. Defaults to the router-side value."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-target-mm-percent",
        type=int,
        default=None,
        help=(
            "Target multimodal arrival percentage where the configured burst "
            "percent is left unchanged. Defaults to 50."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-gain-per-percent",
        type=int,
        default=None,
        help=(
            "Integer gain applied to recent_mm_percent - target_mm_percent when "
            "computing the effective burst percentage. Defaults to 2."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-min-burst-percent",
        type=int,
        default=None,
        help="Lower clamp for dynamic effective burst percentage. Defaults to 0.",
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-max-burst-percent",
        type=int,
        default=None,
        help="Upper clamp for dynamic effective burst percentage. Defaults to 100.",
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-dynamic-smoothing-windows",
        type=int,
        default=None,
        help=(
            "EMA smoothing factor in completed admission windows for dynamic "
            "multimodal share. Defaults to 4."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-backlog-admission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental dynamic role-state policy: when router-local "
            "multimodal prefill backlog pressure reaches a high watermark, "
            "force the current request to use --hybrid-epd-prefill-worker-burst-limit. "
            "When pressure falls, use the normal base/dynamic worker limit."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-prefill-worker-backlog-high-watermark-percent",
        type=int,
        default=None,
        help=(
            "Backlog occupancy percentage that triggers backlog-based prefill "
            "worker burst admission. Defaults to 80 in the router."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-mm-decode-backlog-guard-max-requests",
        type=int,
        default=None,
        help=(
            "Experimental hybrid E/P/D guard: when router-local multimodal "
            "decode streams in flight are at or above this request count, "
            "suppress prefill worker burst admission for new multimodal "
            "prefill requests."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-decode-block-pressure-guard-percent",
        type=int,
        default=None,
        help=(
            "Experimental hybrid E/P/D guard: compute per-decode-worker "
            "block pressure from active_decode_blocks/kv_used_blocks over "
            "total_kv_blocks; when too few decode workers are below this "
            "percentage, suppress prefill worker burst admission."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-decode-block-pressure-guard-min-available-workers",
        type=int,
        default=None,
        help=(
            "Minimum number of decode workers whose block pressure must be "
            "below --hybrid-epd-decode-block-pressure-guard-percent. Defaults "
            "to 1 in the router when the block-pressure guard is enabled."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-decode-request-load-guard-max",
        type=int,
        default=None,
        help=(
            "Experimental hybrid E/P/D guard: compute per-decode-worker "
            "scheduler request load from vLLM ForwardPassMetrics "
            "scheduled_decode_requests + queued_decode_requests; when too few "
            "decode workers are at or below this load, suppress prefill worker "
            "burst admission."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-decode-request-load-guard-min-available-workers",
        type=int,
        default=None,
        help=(
            "Minimum number of decode workers whose ForwardPassMetrics decode "
            "request load must be at or below "
            "--hybrid-epd-decode-request-load-guard-max. Defaults to 1 in the "
            "router when the request-load guard is enabled."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-decode-request-load-guard-prefill-worker-limit",
        type=int,
        default=None,
        help=(
            "When the request-load guard triggers, use this prefill worker "
            "limit instead of suppressing burst all the way back to the base "
            "--hybrid-epd-prefill-worker-limit. The router clamps it between "
            "the base and burst limits. Omit to preserve the existing "
            "suppress-to-base behavior."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-aux-prefill-decode-request-load-max",
        type=int,
        default=None,
        help=(
            "Experimental hybrid E/P/D worker filter: when choosing aux-prefill "
            "decode workers, only admit decode workers whose ForwardPassMetrics "
            "scheduled+queued decode request load is at or below this value. "
            "Primary prefill workers remain eligible."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-aux-prefill-include-unknown-decode-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --hybrid-epd-aux-prefill-decode-request-load-max, keep decode "
            "workers with missing FPM load data eligible. Defaults to true to "
            "avoid cold-start starvation before the first FPM sample arrives."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-direct-text-model-name",
        default=None,
        help=(
            "Register this model alias on hybrid EPD decode workers as an "
            "aggregated direct-text entrypoint, and send the text replay stream "
            "to that alias while image stays on the normal P/D model."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-direct-text-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the experimental Rust prefill-router bypass for text-only "
            "requests. Text uses the normal model name but skips remote prefill "
            "and routes through the same decode worker pool as image decode."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-all-capabilities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable experimental hybrid E/P/D all-capabilities mode: every "
            "non-encode worker registers into both prefill and decode pools "
            "while preserving embedding/KV handoff."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-serve-aux-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When hybrid all-capabilities is enabled, let decode workers also "
            "register an auxiliary prefill endpoint. Disable this to protect "
            "decode workers from image prefill/encode admission."
        ),
    )
    parser.add_argument(
        "--hybrid-epd-serve-aux-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When hybrid all-capabilities is enabled, let prefill workers also "
            "register an auxiliary decode endpoint. Disable this to protect "
            "prefill workers from decode drain."
        ),
    )
    parser.add_argument("--encoder-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--encoder-max-concurrent", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--encode-dispatch-concurrency", type=int, default=None)
    parser.add_argument("--eworker-microbatch-max-items", type=int, default=None)
    parser.add_argument("--eworker-microbatch-wait-ms", type=float, default=None)
    parser.add_argument("--eworker-parallel-preprocess", action="store_true")
    parser.add_argument(
        "--eworker-inline-encode",
        action="store_true",
        help=(
            "Run microbatched E-worker image processor and vision encode inline "
            "on the E-worker event loop instead of through an executor. "
            "Experimental: reduces executor resume gaps but can block RPC handling."
        ),
    )
    parser.add_argument(
        "--eworker-microbatch-thread",
        action="store_true",
        help=(
            "Run the E-worker microbatch processor/vision loop in a dedicated "
            "thread. Experimental: keeps the main E asyncio loop available for "
            "RPC/image loading while avoiding executor resume gaps."
        ),
    )
    parser.add_argument("--eworker-torch-num-threads", type=int, default=None)
    parser.add_argument("--eworker-torch-num-interop-threads", type=int, default=None)
    parser.add_argument("--eworker-timing-breakdown", action="store_true")
    parser.add_argument(
        "--eworker-executor-workers",
        type=int,
        default=None,
        help=(
            "Enable a dedicated E-worker executor for image processor and vision "
            "encode calls, isolating them from asyncio's default thread pool."
        ),
    )
    parser.add_argument(
        "--eworker-image-decode-executor-workers",
        type=int,
        default=None,
        help=(
            "Enable a dedicated E-worker executor for base64/PIL image decode "
            "inside ImageLoader. This is CPU decode offload, not GPU preprocessing."
        ),
    )
    parser.add_argument("--mm-torch-num-threads", type=int, default=None)
    parser.add_argument(
        "--mm-executor-workers",
        type=int,
        default=None,
        help=(
            "Override the vLLM renderer multimodal preprocessing executor worker "
            "count via DYN_MM_EXECUTOR_WORKERS. Applies to AGG and any vLLM "
            "worker that runs renderer MM preprocessing."
        ),
    )
    parser.add_argument(
        "--frontend-cpu-affinity",
        default=None,
        help="Optional taskset CPU list for the Dynamo frontend process.",
    )
    parser.add_argument(
        "--encoder-cpu-affinity",
        default=None,
        help="Optional taskset CPU list for encode/E-worker vLLM processes.",
    )
    parser.add_argument(
        "--pd-cpu-affinity",
        default=None,
        help="Optional taskset CPU list for aggregated, prefill, decode, and PD workers.",
    )
    parser.add_argument("--gpu-memory-utilization-e", type=float, default=0.2)
    parser.add_argument("--gpu-memory-utilization-pd", type=float, default=0.55)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=0.5)
    parser.add_argument("--dynamo-embedding-cache-capacity-gb", type=float, default=0.0)
    parser.add_argument(
        "--enable-encoder-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the encode-worker-local image embedding cache. Keep this off "
            "for no-cache profile experiments so blue encode spans represent real "
            "vision encode work."
        ),
    )
    parser.add_argument(
        "--embedding-transfer-mode",
        choices=["local", "nixl-write", "nixl-read"],
        default="nixl-write",
    )
    parser.add_argument("--disable-prefix-caching", action="store_true", default=True)
    parser.add_argument("--settle-seconds", type=float, default=1800.0)
    parser.add_argument("--max-inflight", type=int, default=20000)
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--prewarm-requests-per-stream", type=int, default=0)
    parser.add_argument("--prewarm-after-sleep-s", type=float, default=2.0)
    parser.add_argument("--slo-ttft-ms", type=int, default=20000)
    parser.add_argument("--slo-tpot-ms", type=int, default=100)
    parser.add_argument("--request-timeout-seconds", type=float, default=21600.0)
    parser.add_argument(
        "--forwardpass-metric-port-base",
        type=int,
        default=None,
        help=(
            "Enable vLLM InstrumentedScheduler/FPM by assigning each worker a "
            "DYN_FORWARDPASS_METRIC_PORT range starting at this base. The runner "
            "uses base + worker_offset * 16 to avoid per-process dp_rank=0 port collisions."
        ),
    )
    parser.add_argument("--ucx-tls", default="tcp,cuda_copy,cuda_ipc,self")
    parser.add_argument("--ucx-net-devices", default=None)
    parser.add_argument("--nixl-write-receiver-buffer-gb", type=float, default=None)
    parser.add_argument(
        "--nixl-write-receive-timeout-seconds", type=float, default=None
    )
    parser.add_argument("--nixl-write-receiver-device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.hybrid_epd_role_endpoints
        and args.hybrid_epd_encode_client_roles == "shared"
    ):
        raise ValueError(
            "--hybrid-epd-role-endpoints requires "
            "--hybrid-epd-encode-client-roles such as prefill,decode"
        )
    if args.run_dir is None:
        args.run_dir = ROOT / "runs" / f"{now_slug()}_dynamo_vllm_text_image_qps32_1to1"
    args.run_dir = args.run_dir.resolve()
    args.replay_jsonl = args.replay_jsonl.resolve()
    if not args.replay_jsonl.exists():
        raise FileNotFoundError(args.replay_jsonl)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    collect_manifest(args)
    for topology in split_csv(args.topologies):
        run_topology(args, topology)


if __name__ == "__main__":
    main()
