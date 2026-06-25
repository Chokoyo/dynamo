# Memory-Budget Colocated E/PD Reproduction

This directory preserves the memory-budget AGG vs colocated E/PD benchmark
configuration used for the 35B and 122B Qwen VL experiments.

The runnable script is:

```bash
benchmarks/multimodal/sweep/experiments/memory_budget_coloc_epd/run_memory_budget_coloc_repro.sh
```

The replay harness is copied into the same directory:

```text
run_dynamo_vllm_replay.py
fixed_rate_client.py
```

It intentionally does **not** pass `--nixl-write-receiver-buffer-gb 32` or
`--nixl-write-receiver-device cuda`, and it unsets the corresponding
`DYNAMO_NIXL_WRITE_RECEIVER_*` environment variables. The latest memory-budget
tables had AGG and colocated E/PD peak GPU memory within roughly the same
range, so keeping a 32 GiB CUDA receiver buffer in the repro config would be
misleading and can OOM on a runtime that honors it.

## Scope

| Model | Workloads | OSL | Repeats | Topologies |
| --- | --- | --- | ---: | --- |
| `/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989` | image-only, text+image 1:1 | 128, 512, 1024 | 3 | AGG, colocated E/PD |
| `/models/Qwen3.5-122B-A10B-FP8` | image-only, text+image 1:1 | 128, 512 | 3 | AGG, colocated E/PD |

Fixed replay JSONL inputs are expected under `/workspace/experiments/epd_repro/bench`:

- image-only: `image_only_4x768_out{OSL}_qps20_60s_pool128_replay.jsonl`
- text+image 1:1: `text_image_4x768_out{OSL}_qps20_60s_replay.jsonl`

Set `EPD_REPRO_ROOT` if the harness is not at `/workspace/experiments/epd_repro`.
The script sets `DYNAMO_ROOT` to the current Dynamo checkout unless it is
already provided.

## Common Config

```bash
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
```

## Topologies

AGG:

```bash
--topologies baseline
--baseline-tensor-parallel-size 1
--frontend-cpu-affinity 32-139
--pd-cpu-affinity 32-139
--gpu-memory-utilization-pd 0.90
```

Colocated E/PD:

```bash
--topologies e_pd
--epd-pd-gpus 0,1,2,3
--epd-encoder-gpu-groups "0;0;1;1;2;2;3;3"
--allow-epd-gpu-overlap
--epd-encoder-tensor-parallel-size 1
--epd-pd-tensor-parallel-size 1
--encoder-max-num-batched-tokens 32768
--eworker-executor-workers 2
--eworker-torch-num-threads 16
--encoder-cpu-affinity 0-31
--frontend-cpu-affinity 32-139
--pd-cpu-affinity 32-139
--gpu-memory-utilization-e 0.10
--gpu-memory-utilization-pd 0.88
--no-enable-encoder-cache
--embedding-transfer-mode nixl-write
--ucx-tls rc,tcp,cuda_copy,cuda_ipc,self
--ucx-net-devices mlx5_0:1
```

## vLLM Patch Needed For AGG Fairness

The AGG path performs image preprocessing inside vLLM's renderer. To compare
AGG against an E/PD setup that has tuned E-worker CPU preprocessing, the AGG
path needs an equivalent vLLM renderer tuning surface.

Patch `vllm/renderers/base.py` to:

1. Read `DYN_MM_TORCH_NUM_THREADS`.
2. Read `DYN_MM_EXECUTOR_WORKERS`.
3. Use a dedicated multimodal preprocessing executor when
   `DYN_MM_EXECUTOR_WORKERS` is set.
4. Pass `DYN_MM_TORCH_NUM_THREADS` into vLLM's
   `set_default_torch_num_threads(...)` around multimodal processor creation
   and `mm_processor.apply(...)`.

The essential shape is:

```python
self._mm_torch_num_threads = _positive_int_env("DYN_MM_TORCH_NUM_THREADS")
mm_executor_workers = _positive_int_env("DYN_MM_EXECUTOR_WORKERS")
if mm_executor_workers is not None:
    self._mm_executor = ThreadPoolExecutor(max_workers=mm_executor_workers)
else:
    self._mm_executor = self._executor

with set_default_torch_num_threads(self._mm_torch_num_threads):
    mm_inputs = mm_processor.apply(mm_processor_inputs, mm_timing_ctx)
```

Without this vLLM patch, setting only `torch.set_num_threads(...)` in Dynamo is
not enough: vLLM wraps multimodal processor work in
`set_default_torch_num_threads()`, which uses `OMP_NUM_THREADS` or defaults to
1. Setting `OMP_NUM_THREADS` in Dynamo can tune torch threads coarsely, but it
does not provide a multimodal-only executor control.

## Why E/PD Does Not Need That vLLM Renderer Patch

The E/PD image encode path uses Dynamo's encode worker, not the AGG vLLM
renderer image preprocessing path. E-worker tuning is implemented in Dynamo's
`EncodeWorkerHandler` via:

```text
DYN_EWORKER_TORCH_NUM_THREADS
DYN_EWORKER_EXECUTOR_WORKERS
```

Those controls set `torch.set_num_threads(...)` and the E-worker encode executor
inside the standalone encode worker. PD workers receive precomputed embeddings
through the embedding handoff path. Therefore the E-worker optimization does not
depend on the vLLM renderer patch above.

The script does not override encode dispatch batch size. With 4 images per
request and 8 colocated E workers, Dynamo's existing default dispatch rule
already gives an effective encode batch size of 1:
`max(1, image_count // encode_worker_count) = max(1, 4 // 8) = 1`.

The vLLM renderer patch is still useful for AGG fairness, and for any fallback
or non-E-worker path that performs multimodal preprocessing inside vLLM.

## Commands

Run only 35B:

```bash
MODEL_SET=35b ./run_memory_budget_coloc_repro.sh
```

Run only 122B:

```bash
MODEL_SET=122b ./run_memory_budget_coloc_repro.sh
```

Run both:

```bash
MODEL_SET=both ./run_memory_budget_coloc_repro.sh
```

Override the run id:

```bash
SWEEP_ID=20260624T_memory_budget_coloc_repro_rerun MODEL_SET=35b \
  ./run_memory_budget_coloc_repro.sh
```
