#!/usr/bin/env bash
set -u

# Reproduce the memory-budget AGG vs colocated E/PD tables without the
# misleading 32 GiB NIXL write receiver buffer override.
#
# Usage:
#   MODEL_SET=35b  ./run_memory_budget_coloc_repro.sh
#   MODEL_SET=122b ./run_memory_budget_coloc_repro.sh
#   MODEL_SET=both ./run_memory_budget_coloc_repro.sh
#
# This script lives in the Dynamo tree and uses the local replay harness copy in
# this directory. It reuses replay JSONL inputs and output directories under
# /workspace/experiments/epd_repro. Set EPD_REPRO_ROOT to override that path.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DYNAMO_CHECKOUT=$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)

ROOT=${EPD_REPRO_ROOT:-/workspace/experiments/epd_repro}
REPLAY_SCRIPT=${REPLAY_SCRIPT:-$SCRIPT_DIR/run_dynamo_vllm_replay.py}
export DYNAMO_ROOT=${DYNAMO_ROOT:-$DYNAMO_CHECKOUT}
export DYNAMO_RUNTIME_ROOT=${DYNAMO_RUNTIME_ROOT:-$DYNAMO_ROOT}
export VLLM_SRC=${VLLM_SRC:-/workspace/vllm-src}
export PATH="$ROOT/bin:$PATH"
export ETCD_UNSUPPORTED_ARCH=arm64
export PYTHONPATH="$ROOT/python_dist:/workspace/deps/humming:${PYTHONPATH:-}"

# Do not inherit a large NIXL receiver buffer from the shell. The tables this
# script reproduces did not reflect a 32 GiB CUDA receiver buffer.
unset DYNAMO_NIXL_WRITE_RECEIVER_BUFFER_GB
unset DYNAMO_NIXL_WRITE_RECEIVER_BUFFER_BYTES
unset DYNAMO_NIXL_WRITE_RECEIVER_DEVICE

MODEL_35B=/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989
MODEL_122B=/models/Qwen3.5-122B-A10B-FP8

MODEL_SET=${MODEL_SET:-both}
SWEEP_ID=${SWEEP_ID:-20260624T_memory_budget_coloc_repro}
STATUS=$ROOT/runs/${SWEEP_ID}_status.tsv

mkdir -p "$ROOT/runs"
printf "case\tstatus\n" > "$STATUS"

COMMON_ARGS=(
  --frontend-router-mode random
  --max-model-len 8192
  --max-num-seqs 64
  --images-per-request 4
  --mm-torch-num-threads 16
  --mm-executor-workers 1
  --disable-prefix-caching
  --mm-processor-cache-gb 0
  --dynamo-embedding-cache-capacity-gb 0
  --settle-seconds 1800
  --max-inflight 20000
  --num-warmups 3
  --prewarm-requests-per-stream 24
  --prewarm-after-sleep-s 3
  --slo-ttft-ms 20000
  --slo-tpot-ms 100
  --request-timeout-seconds 1800
)

run_case() {
  local name=$1
  shift
  printf "START\t%s\n" "$name"
  if python "$REPLAY_SCRIPT" "$@"; then
    printf "%s\tok\n" "$name" >> "$STATUS"
    printf "DONE\t%s\n" "$name"
  else
    printf "%s\tfailed\n" "$name" >> "$STATUS"
    printf "FAILED\t%s\n" "$name"
  fi
}

port_idx=0
next_ports() {
  HTTP_PORT=$((29400 + port_idx * 20))
  SYSTEM_PORT_BASE=$((29500 + port_idx * 20))
  ETCD_CLIENT_PORT=$((14000 + port_idx * 2))
  ETCD_PEER_PORT=$((14001 + port_idx * 2))
  METRIC_PORT_BASE=$((48200 + port_idx * 20))
  port_idx=$((port_idx + 1))
}

replay_path() {
  local osl=$1
  local workload=$2
  if [[ "$workload" == "image" ]]; then
    printf "%s/bench/image_only_4x768_out%s_qps20_60s_pool128_replay.jsonl" "$ROOT" "$osl"
  else
    printf "%s/bench/text_image_4x768_out%s_qps20_60s_replay.jsonl" "$ROOT" "$osl"
  fi
}

stream_arg() {
  local workload=$1
  if [[ "$workload" == "image" ]]; then
    printf "image"
  else
    printf "text,image"
  fi
}

workload_arg() {
  local workload=$1
  if [[ "$workload" == "image" ]]; then
    printf "image_only_qps20"
  else
    printf "text_image_qps20_1to1"
  fi
}

run_agg_low() {
  local model_label=$1
  local model_path=$2
  local rep=$3
  local osl=$4
  local workload=$5
  local replay streams workload_name
  replay=$(replay_path "$osl" "$workload")
  streams=$(stream_arg "$workload")
  workload_name=$(workload_arg "$workload")
  next_ports
  run_case "${model_label}_r${rep}_osl${osl}_${workload}_agg_pd0p90" \
    --run-dir "$ROOT/runs/${SWEEP_ID}_${model_label}_r${rep}_osl${osl}_${workload}_agg_pd0p90" \
    --topologies baseline \
    --model "$model_path" \
    --replay-jsonl "$replay" \
    --replay-workload "$workload_name" \
    --replay-client-streams "$streams" \
    --http-port "$HTTP_PORT" \
    --system-port-base "$SYSTEM_PORT_BASE" \
    --etcd-client-port "$ETCD_CLIENT_PORT" \
    --etcd-peer-port "$ETCD_PEER_PORT" \
    --forwardpass-metric-port-base "$METRIC_PORT_BASE" \
    --baseline-tensor-parallel-size 1 \
    --frontend-cpu-affinity 32-139 \
    --pd-cpu-affinity 32-139 \
    --gpu-memory-utilization-e 0.10 \
    --gpu-memory-utilization-pd 0.90 \
    "${COMMON_ARGS[@]}"
}

run_epd_colocate_low() {
  local model_label=$1
  local model_path=$2
  local rep=$3
  local osl=$4
  local workload=$5
  local replay streams workload_name
  replay=$(replay_path "$osl" "$workload")
  streams=$(stream_arg "$workload")
  workload_name=$(workload_arg "$workload")
  next_ports
  run_case "${model_label}_r${rep}_osl${osl}_${workload}_epd_coloc_pd0p88_e0p10" \
    --run-dir "$ROOT/runs/${SWEEP_ID}_${model_label}_r${rep}_osl${osl}_${workload}_epd_coloc_pd0p88_e0p10" \
    --topologies e_pd \
    --model "$model_path" \
    --replay-jsonl "$replay" \
    --replay-workload "$workload_name" \
    --replay-client-streams "$streams" \
    --http-port "$HTTP_PORT" \
    --system-port-base "$SYSTEM_PORT_BASE" \
    --etcd-client-port "$ETCD_CLIENT_PORT" \
    --etcd-peer-port "$ETCD_PEER_PORT" \
    --forwardpass-metric-port-base "$METRIC_PORT_BASE" \
    --epd-pd-gpus 0,1,2,3 \
    --epd-encoder-gpu-groups "0;0;1;1;2;2;3;3" \
    --allow-epd-gpu-overlap \
    --epd-encoder-tensor-parallel-size 1 \
    --epd-pd-tensor-parallel-size 1 \
    --encoder-max-num-batched-tokens 32768 \
    --eworker-executor-workers 2 \
    --eworker-torch-num-threads 16 \
    --encoder-cpu-affinity 0-31 \
    --frontend-cpu-affinity 32-139 \
    --pd-cpu-affinity 32-139 \
    --gpu-memory-utilization-e 0.10 \
    --gpu-memory-utilization-pd 0.88 \
    --no-enable-encoder-cache \
    --embedding-transfer-mode nixl-write \
    --ucx-tls rc,tcp,cuda_copy,cuda_ipc,self \
    --ucx-net-devices mlx5_0:1 \
    "${COMMON_ARGS[@]}"
}

run_matrix() {
  local model_label=$1
  local model_path=$2
  local reps=$3
  local osls=$4

  for rep in $(seq 1 "$reps"); do
    for osl in $osls; do
      for workload in image text_image; do
        run_agg_low "$model_label" "$model_path" "$rep" "$osl" "$workload"
        run_epd_colocate_low "$model_label" "$model_path" "$rep" "$osl" "$workload"
      done
    done
  done
}

case "$MODEL_SET" in
  35b)
    run_matrix 35b "$MODEL_35B" 3 "128 512 1024"
    ;;
  122b)
    run_matrix 122b "$MODEL_122B" 3 "128 512"
    ;;
  both)
    run_matrix 35b "$MODEL_35B" 3 "128 512 1024"
    run_matrix 122b "$MODEL_122B" 3 "128 512"
    ;;
  *)
    echo "MODEL_SET must be one of: 35b, 122b, both" >&2
    exit 2
    ;;
esac

printf "STATUS\t%s\n" "$STATUS"
