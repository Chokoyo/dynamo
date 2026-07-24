#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen2-vl}"
IMAGE_URL="${IMAGE_URL:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg}"
LOCAL_IMAGE_PATH="${LOCAL_IMAGE_PATH:-}"
MEDIA_PORT="${MEDIA_PORT:-18084}"
RESULT_DIR="${RESULT_DIR:-/mmkv/results}"
LOG_DIR="${LOG_DIR:-/mmkv/logs}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-4}"
WARMUP="${WARMUP:-1}"
SOURCE_KIND="${SOURCE_KIND:-E_COMPUTE}"
SOURCE_KINDS="${SOURCE_KINDS:-}"
IMAGE_COUNT="${IMAGE_COUNT:-1}"
ENCODE_CACHE_CAPACITY_GB="${ENCODE_CACHE_CAPACITY_GB:-2}"
PREFILL_CACHE_CAPACITY_GB="${PREFILL_CACHE_CAPACITY_GB:-2}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
ENCODE_GPU="${ENCODE_GPU:-0}"
ENCODE_GPUS="${ENCODE_GPUS:-$ENCODE_GPU}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
DYNAMO_RUNTIME_PYTHONPATH="${DYNAMO_RUNTIME_PYTHONPATH:-}"
TMPDIR="${TMPDIR:-/mmkv/tmp}"

STORE_DIR="${DYN_FILE_KV:-/mmkv/dynamo-store/sglang-epd-$$}"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$TMPDIR" "$(dirname "$STORE_DIR")"
rm -rf "$STORE_DIR"
mkdir -p "$STORE_DIR"

DYNAMO_SOURCE_PATH="${DYNAMO_SOURCE_PATH-/workspace/components/src}"
export PYTHONPATH="/workspace${DYNAMO_SOURCE_PATH:+:$DYNAMO_SOURCE_PATH}${DYNAMO_RUNTIME_PYTHONPATH:+:$DYNAMO_RUNTIME_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="$STORE_DIR"
export DYN_SGL_EMBEDDING_TRANSFER_MODE=local
export PYTHONUNBUFFERED=1
export RUST_LOG="${RUST_LOG:-info}"

pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${pids[@]}" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

if [[ -n "$LOCAL_IMAGE_PATH" ]]; then
    python3 -m http.server "$MEDIA_PORT" \
        --bind 127.0.0.1 \
        --directory "$(dirname "$LOCAL_IMAGE_PATH")" \
        >"$LOG_DIR/media-server.log" 2>&1 &
    pids+=("$!")
    IMAGE_URL="http://127.0.0.1:${MEDIA_PORT}/$(basename "$LOCAL_IMAGE_PATH")"
    sleep 1
fi

common_args=(
    --enable-multimodal
    --dedicated-mm-encoder
    --model-path "$MODEL_NAME"
    --chat-template "$CHAT_TEMPLATE"
    --page-size 16
    --tp 1
    --trust-remote-code
    --skip-tokenizer-init
    --host 0.0.0.0
    --disaggregation-transfer-backend nixl
    --max-running-requests 64
)

if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
    common_args+=(--disable-radix-cache)
fi

echo "Starting decode worker on GPU $DECODE_GPU"
DYN_SYSTEM_PORT=18083 CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
python3 -m dynamo.sglang \
    "${common_args[@]}" \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 12345 \
    --nccl-port 29503 \
    --mem-fraction-static 0.75 \
    --max-total-tokens 16384 \
    >"$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")

sleep 5
IFS=',' read -r -a encode_gpus <<<"$ENCODE_GPUS"
if [[ "${#encode_gpus[@]}" -eq 0 ]]; then
    echo "ENCODE_GPUS must contain at least one GPU" >&2
    exit 1
fi
for encode_index in "${!encode_gpus[@]}"; do
    encode_log="$LOG_DIR/encode.log"
    if [[ "$encode_index" -gt 0 ]]; then
        encode_log="$LOG_DIR/encode-$encode_index.log"
    fi
    echo "Starting encode worker $encode_index on GPU ${encode_gpus[$encode_index]}"
    DYN_SYSTEM_PORT=$((18100 + encode_index)) \
    CUDA_VISIBLE_DEVICES="${encode_gpus[$encode_index]}" \
    python3 -m dynamo.sglang \
        --enable-multimodal \
        --disaggregation-mode encode \
        --model-path "$MODEL_NAME" \
        --chat-template "$CHAT_TEMPLATE" \
        --skip-tokenizer-init \
        --multimodal-embedding-cache-capacity-gb "$ENCODE_CACHE_CAPACITY_GB" \
        --multimodal-embedding-cache-publisher \
        >"$encode_log" 2>&1 &
    pids+=("$!")
done

sleep 5
echo "Starting prefill worker on GPU $PREFILL_GPU"
DYN_SYSTEM_PORT=18082 CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
python3 -m dynamo.sglang \
    "${common_args[@]}" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 12345 \
    --nccl-port 29502 \
    --mem-fraction-static 0.35 \
    --max-total-tokens 8192 \
    --multimodal-embedding-cache-capacity-gb "$PREFILL_CACHE_CAPACITY_GB" \
    --multimodal-embedding-cache-publisher \
    >"$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")

source_label="${SOURCE_KIND,,}"
source_args=(--source-kind "$SOURCE_KIND")
if [[ -n "$SOURCE_KINDS" ]]; then
    source_label=mixed
    source_args=(--source-kinds "$SOURCE_KINDS")
fi
result_file="$RESULT_DIR/sglang-epd-${source_label}-c${CONCURRENCY//,/-}.json"
python3 -m benchmarks.multimodal.sglang_epd_object_benchmark \
    --model "$MODEL_NAME" \
    --image-url "$IMAGE_URL" \
    --image-count "$IMAGE_COUNT" \
    --concurrency "$CONCURRENCY" \
    --requests-per-level "$REQUESTS_PER_LEVEL" \
    --warmup "$WARMUP" \
    "${source_args[@]}" \
    --timeout 600 \
    --json-output "$result_file" \
    | tee "$LOG_DIR/benchmark.log"

sleep 1
python3 - "$result_file" "$LOG_DIR" "$WARMUP" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
warmup = int(sys.argv[3])
payload = json.loads(result_path.read_text(encoding="utf-8"))
image_count = int(payload["image_count"])
source_kinds = payload["source_kinds"]
encode_worker_count = len(payload["workers"]["encode"])
remote_worker_slots = {
    object_index % encode_worker_count
    for object_index, source_kind in enumerate(source_kinds)
    if source_kind != "P_LOCAL"
}
remote_dispatches_per_request = len(remote_worker_slots)
warm_worker_slots = {
    object_index % encode_worker_count
    for object_index, source_kind in enumerate(source_kinds)
    if source_kind == "E_CACHE"
}
if "P_LOCAL" in source_kinds:
    warm_worker_slots.add(0)
warm_dispatches = len(warm_worker_slots)
measured_requests = sum(int(row["requests"]) for row in payload["results"])
level_warmups = warmup * len(payload["results"])
cache_warmup = int(bool(warm_worker_slots))
final_requests = measured_requests + level_warmups
total_requests = final_requests + cache_warmup

logs = {
    name: (log_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    for name in ("decode", "prefill")
}
logs["encode"] = "\n".join(
    path.read_text(encoding="utf-8", errors="replace")
    for path in sorted(log_dir.glob("encode*.log"))
)
observed = {
    "decode_direct": logs["decode"].count(
        "SGLang EPD decode routing request to prefill worker"
    ),
    "encode_process": logs["encode"].count(
        "SGLang EPD encode worker processing object indices"
    ),
    "encode_cache_hit": logs["encode"].count(
        "Embedding cache hit for IMAGE URL index"
    ),
    "prefill_cache_hit": logs["prefill"].count(
        "SGLang EPD prefill cache hit for object"
    ),
    "prefill_received": logs["prefill"].count(
        "SGLang EPD prefill received objects"
    ),
}
expected = {
    "decode_direct": total_requests,
    "encode_process": warm_dispatches + final_requests * remote_dispatches_per_request,
    "encode_cache_hit": final_requests * source_kinds.count("E_CACHE"),
    "prefill_cache_hit": final_requests * source_kinds.count("P_LOCAL"),
    "prefill_received": warm_dispatches
    + final_requests * remote_dispatches_per_request,
}

if observed != expected:
    raise RuntimeError(
        f"SGLang EPD path audit failed for {source_kinds}: "
        f"expected={expected}, observed={observed}"
    )
payload["audit"] = {
    "expected": expected,
    "observed": observed,
    "measured_requests": measured_requests,
    "level_warmups": level_warmups,
    "cache_warmup": cache_warmup,
    "warm_dispatches": warm_dispatches,
    "remote_dispatches_per_request": remote_dispatches_per_request,
}
result_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload["audit"], indent=2, sort_keys=True))
PY

echo "Routing evidence:"
grep -hE "SGLang EPD (decode routing|prefill (received|cache hit)|encode worker processing)|Embedding cache hit" \
    "$LOG_DIR"/{decode,prefill}.log "$LOG_DIR"/encode*.log || true
echo "Result: $result_file"
