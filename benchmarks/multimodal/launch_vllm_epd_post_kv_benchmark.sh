#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
LOCAL_IMAGE_PATH="${LOCAL_IMAGE_PATH:?LOCAL_IMAGE_PATH is required}"
RESULT_DIR="${RESULT_DIR:-/mmkv/results/vllm-post-kv}"
LOG_DIR="${LOG_DIR:-/mmkv/logs/vllm-post-kv}"
MEDIA_PORT="${MEDIA_PORT:-18094}"
ENCODE_GPU="${ENCODE_GPU:-0}"
PD_GPU="${PD_GPU:-1}"
CONCURRENCY="${CONCURRENCY:-1,8,32}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-64}"
DYNAMO_RUNTIME_PYTHONPATH="${DYNAMO_RUNTIME_PYTHONPATH:-/mmkv/dynamo-current}"
TMPDIR="${TMPDIR:-/mmkv/tmp}"

STORE_DIR="${DYN_FILE_KV:-/mmkv/dynamo-store/vllm-post-kv-$$}"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$TMPDIR" "$(dirname "$STORE_DIR")"
rm -rf "$STORE_DIR"
mkdir -p "$STORE_DIR"

export PYTHONPATH="/workspace/components/src:$DYNAMO_RUNTIME_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="$STORE_DIR"
export DYN_MULTIMODAL_EPD_ROUTING_MODE=enforce
export DYN_MULTIMODAL_EPD_AUDIT_DIR="$RESULT_DIR/audit"
export DYN_VLLM_EMBEDDING_TRANSFER_MODE=local
export DYN_MM_ALLOW_INTERNAL=1
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export RUST_LOG="${RUST_LOG:-info}"
rm -rf "$DYN_MULTIMODAL_EPD_AUDIT_DIR"
mkdir -p "$DYN_MULTIMODAL_EPD_AUDIT_DIR"

pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill -- "-$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 20); do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if [[ "$alive" -eq 0 ]]; then
            break
        fi
        sleep 0.5
    done
    for pid in "${pids[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
    done
    wait "${pids[@]}" 2>/dev/null || true
    rm -rf "$STORE_DIR"
    exit "$status"
}
trap cleanup EXIT INT TERM

setsid python3 -m http.server "$MEDIA_PORT" \
    --bind 127.0.0.1 \
    --directory "$(dirname "$LOCAL_IMAGE_PATH")" \
    >"$LOG_DIR/media-server.log" 2>&1 &
pids+=("$!")
IMAGE_URL="http://127.0.0.1:${MEDIA_PORT}/$(basename "$LOCAL_IMAGE_PATH")"

setsid env DYN_SYSTEM_PORT=18091 VLLM_USE_V2_MODEL_RUNNER=0 \
CUDA_VISIBLE_DEVICES="$ENCODE_GPU" python3 -m dynamo.vllm \
    --enable-multimodal \
    --disaggregation-mode encode \
    --embedding-transfer-mode local \
    --multimodal-embedding-cache-publisher \
    --model "$MODEL_NAME" \
    --enforce-eager \
    --gpu-memory-utilization 0.3 \
    >"$LOG_DIR/encode.log" 2>&1 &
pids+=("$!")

setsid env DYN_SYSTEM_PORT=18092 VLLM_USE_V2_MODEL_RUNNER=0 \
CUDA_VISIBLE_DEVICES="$PD_GPU" python3 -m dynamo.vllm \
    --route-to-encoder \
    --enable-multimodal \
    --enable-mm-embeds \
    --disaggregation-mode pd \
    --embedding-transfer-mode local \
    --multimodal-embedding-cache-capacity-gb 2 \
    --multimodal-embedding-cache-publisher \
    --model "$MODEL_NAME" \
    --enforce-eager \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    >"$LOG_DIR/pd.log" 2>&1 &
pids+=("$!")

sleep 2
python3 -m benchmarks.multimodal.vllm_epd_post_kv_benchmark \
    --model "$MODEL_NAME" \
    --image-url "$IMAGE_URL" \
    --concurrency "$CONCURRENCY" \
    --requests-per-level "$REQUESTS_PER_LEVEL" \
    --timeout 600 \
    --audit-dir "$DYN_MULTIMODAL_EPD_AUDIT_DIR" \
    --json-output "$RESULT_DIR/result.json" \
    | tee "$LOG_DIR/benchmark.log"

echo "Routing evidence:"
grep -hE "vLLM (EPD encode|post-KV EPD)" "$LOG_DIR"/{encode,pd}.log || true
echo "Result: $RESULT_DIR/result.json"
