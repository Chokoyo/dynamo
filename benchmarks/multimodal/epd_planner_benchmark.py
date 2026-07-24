# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic CPU benchmark for the multimodal joint KV/EC planner."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from dynamo.common.multimodal_epd import (
    EmbeddingNamespace,
    EncodeCandidate,
    JointMMPlanner,
    MMCacheResidencyEvent,
    MMCacheResidencyIndex,
    MMObjectRef,
    MMTokenSpan,
    ObjectSourceKind,
    PrefillCandidate,
    ResidencyAction,
    WorkerRole,
)


def build_scenario(
    *, object_count: int, ec_hit_ratio: float, prefill_count: int, encoder_count: int
) -> tuple[
    JointMMPlanner,
    EmbeddingNamespace,
    list[MMObjectRef],
    list[PrefillCandidate],
    list[EncodeCandidate],
]:
    namespace = EmbeddingNamespace.from_processor_config(
        model_id="benchmark/model",
        model_revision="main",
        processor_config={"min_pixels": 256, "max_pixels": 1280},
    )
    objects = []
    for object_index in range(object_count):
        content_id = f"image-{object_index}"
        objects.append(
            MMObjectRef(
                object_index=object_index,
                modality="image",
                content_id=content_id,
                embedding_cache_key=namespace.cache_key(content_id),
                model_token_spans=(
                    MMTokenSpan(object_index * 256, (object_index + 1) * 256),
                ),
                source_kind=ObjectSourceKind.URL,
            )
        )

    index = MMCacheResidencyIndex(ttl_seconds=300)
    hit_count = round(object_count * ec_hit_ratio)
    for object_index, obj in enumerate(objects[:hit_count]):
        worker_id = 100 + object_index % encoder_count
        index.apply(
            MMCacheResidencyEvent(
                worker_id=worker_id,
                worker_generation=1,
                worker_role=WorkerRole.ENCODE,
                namespace=namespace,
                cache_key=obj.embedding_cache_key,
                action=ResidencyAction.ADD,
                bytes=4 * 1024 * 1024,
                observed_at=1.0,
            )
        )

    prefill_candidates = [
        PrefillCandidate(
            worker_id=worker_id,
            worker_generation=1,
            predicted_saved_prefill_ms=float((worker_id % 4) * 2),
            queue_delay_ms=float(worker_id % 3) * 0.2,
        )
        for worker_id in range(prefill_count)
    ]
    encode_candidates = [
        EncodeCandidate(
            worker_id=100 + worker_id,
            worker_generation=1,
            queue_delay_ms=float(worker_id % 3) * 0.1,
            encode_ms_by_key={obj.embedding_cache_key: 2.5 for obj in objects},
        )
        for worker_id in range(encoder_count)
    ]
    return (
        JointMMPlanner(index),
        namespace,
        objects,
        prefill_candidates,
        encode_candidates,
    )


def run_case(
    *,
    concurrency: int,
    object_count: int,
    ec_hit_ratio: float,
    iterations: int,
) -> dict[str, float | int]:
    planner, namespace, objects, prefill_candidates, encode_candidates = build_scenario(
        object_count=object_count,
        ec_hit_ratio=ec_hit_ratio,
        prefill_count=4,
        encoder_count=4,
    )
    samples_us: list[float] = []
    plans = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        for _request in range(concurrency):
            plans.append(
                planner.plan(
                    namespace=namespace,
                    objects=objects,
                    prefill_candidates=prefill_candidates,
                    encode_candidates=encode_candidates,
                    now=2.0,
                )
            )
        elapsed_ns = time.perf_counter_ns() - start
        samples_us.append(elapsed_ns / concurrency / 1_000)

    ordered = sorted(samples_us)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "concurrency": concurrency,
        "objects": object_count,
        "ec_hit_ratio": ec_hit_ratio,
        "iterations": iterations,
        "median_us_per_plan": statistics.median(samples_us),
        "p95_us_per_plan": ordered[p95_index],
        "selected_prefill": plans[-1].target_p_worker_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    results = [
        run_case(
            concurrency=concurrency,
            object_count=object_count,
            ec_hit_ratio=ec_hit_ratio,
            iterations=args.iterations,
        )
        for concurrency in (1, 8, 32)
        for object_count in (1, 2, 4)
        for ec_hit_ratio in (0.0, 0.5, 1.0)
    ]
    payload = {
        "benchmark": "multimodal_epd_joint_planner",
        "results": results,
        "summary": {
            "max_p95_us_per_plan": max(result["p95_us_per_plan"] for result in results),
            "cases": len(results),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
