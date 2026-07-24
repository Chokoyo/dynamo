# Joint KV and Embedding-Cache-Aware EPD Routing

This directory contains the validation harnesses for Dynamo's joint multimodal
EPD routing policy. The policy combines prefill KV overlap, embedding-cache
residency, worker load, transfer cost, and multi-worker fanout cost when it
selects a prefill worker and an ordered source for every image object.

## Runtime modes

Set `DYN_MULTIMODAL_EPD_ROUTING_MODE` on the frontend:

- `off` preserves the legacy request-level path.
- `observe` records the selected target P and object plan without requiring
  workers to enforce the plan.
- `enforce` sends the selected target P and object plan to compatible workers.

Malformed, incomplete, unsupported, or stale plans fail closed to the legacy
path. Workers validate object indices, cache keys, worker generations, transfer
metadata, and shapes before model execution. UUID-only objects fail explicitly
when no advertised copy remains.

## Backend capability boundary

The enforced object-aware path currently supports image objects.

- vLLM resolves remaining embeddings after its authoritative KV lookup. A full
  KV hit performs no encoder dispatch or embedding transfer. P-local cache hits
  also avoid remote transfer.
- SGLang coordinates P-local and multi-E object resolution before its
  authoritative radix lookup. The radix cache still prevents KV-covered rows
  from entering model forward, but phase-one SGLang may eagerly encode or
  transfer embeddings that the radix cache later proves unnecessary.
- Supported video requests remain on each backend's existing EPD path. The
  aware router does not construct or enforce a video object plan until an exact,
  model-specific sampled-video identity and metadata contract is available.
  This is a fail-closed compatibility path, not video-aware routing.

## Benchmarks

The CPU planner microbenchmark measures planning overhead across C1, C8, and
C32, one to four objects, and 0%, 50%, and 100% EC hit ratios:

```bash
PYTHONPATH=components/src \
python3 benchmarks/multimodal/epd_planner_benchmark.py \
  --json-output /tmp/epd-planner.json
```

The SGLang launcher validates `P_LOCAL`, `E_CACHE`, `E_COMPUTE`, and mixed
per-object plans. `SOURCE_KINDS` accepts a comma-separated ordered plan, and
`ENCODE_GPUS` accepts multiple encode workers:

```bash
MODEL_NAME=/path/to/model \
LOCAL_IMAGE_PATH=/path/to/image.jpg \
SOURCE_KINDS=P_LOCAL,E_COMPUTE,E_COMPUTE,E_COMPUTE \
ENCODE_GPUS=0,1 PD_GPU=2 \
CONCURRENCY=1,8,32 REQUESTS_PER_LEVEL=32 \
bash benchmarks/multimodal/launch_sglang_epd_object_benchmark.sh
```

The vLLM launcher validates cold E compute, P-local EC reuse, and full-KV
post-lookup suppression:

```bash
MODEL_NAME=/path/to/model \
LOCAL_IMAGE_PATH=/path/to/image.jpg \
ENCODE_GPU=0 PD_GPU=1 \
CONCURRENCY=1,8,32 REQUESTS_PER_LEVEL=32 \
bash benchmarks/multimodal/launch_vllm_epd_post_kv_benchmark.sh
```

Both GPU harnesses emit JSON results plus audit counts. A run is invalid if its
expected and observed audit counts differ or if output token parity fails.

## H100 validation snapshot

On July 24, 2026, Qwen2.5-VL-3B on two H100 GPUs produced output token IDs
`[785, 151645]` for every measured SGLang C1/C8/C32 case. At C32, P-local EC
reached 91.24 requests/s, E-cache reached 74.38 requests/s, and cold E-compute
reached 35.30 requests/s. A four-image mixed request successfully combined one
P-local object with three objects split across two E workers.

The matching vLLM run showed the post-KV gate directly: at C32, full-KV hits
performed zero encoder dispatches and reached 165.26 requests/s, P-local EC
reached 142.61 requests/s, and cold E-compute reached 17.07 requests/s. These
numbers are a validation snapshot, not portable production defaults; retune the
cost weights for the deployed model, media distribution, and worker topology.
