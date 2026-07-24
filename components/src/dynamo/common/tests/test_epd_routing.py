# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.multimodal_epd import (
    EmbeddingNamespace,
    EncodeCandidate,
    JointMMPlanner,
    MMCacheResidencyEvent,
    MMCacheResidencyIndex,
    MMObjectRef,
    MMSourceKind,
    MMTokenSpan,
    ObjectSourceKind,
    PrefillCandidate,
    ResidencyAction,
    WorkerRole,
    canonical_wire_json,
    to_wire,
)


def _namespace(**processor_overrides) -> EmbeddingNamespace:
    return EmbeddingNamespace.from_processor_config(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision="abc123",
        processor_config={"min_pixels": 256, **processor_overrides},
    )


def _object(
    namespace: EmbeddingNamespace,
    index: int,
    *,
    source_kind: ObjectSourceKind = ObjectSourceKind.URL,
) -> MMObjectRef:
    content_id = f"image-{index}"
    return MMObjectRef(
        object_index=index,
        modality="image",
        content_id=content_id,
        embedding_cache_key=namespace.cache_key(content_id),
        model_token_spans=(MMTokenSpan(index * 16, index * 16 + 16),),
        source_kind=source_kind,
    )


def _add(
    index: MMCacheResidencyIndex,
    namespace: EmbeddingNamespace,
    obj: MMObjectRef,
    *,
    worker_id: int,
    generation: int,
    role: WorkerRole,
    observed_at: float = 10.0,
) -> None:
    index.apply(
        MMCacheResidencyEvent(
            worker_id=worker_id,
            worker_generation=generation,
            worker_role=role,
            namespace=namespace,
            cache_key=obj.embedding_cache_key,
            action=ResidencyAction.ADD,
            bytes=100_000,
            observed_at=observed_at,
        )
    )


def test_namespace_scopes_processor_configuration() -> None:
    first = _namespace(min_pixels=256)
    same = _namespace(min_pixels=256)
    different = _namespace(min_pixels=512)
    assert first == same
    assert first.processor_fingerprint != different.processor_fingerprint
    assert first.cache_key("media") != different.cache_key("media")


def test_cross_language_namespace_golden_vector() -> None:
    namespace = EmbeddingNamespace.from_processor_config(
        model_id="example/model",
        model_revision="revision-1",
        processor_config={"nested": {"b": 2, "a": 1}, "max_pixels": 1280},
    )
    assert namespace.processor_fingerprint == "f445a889cca07df19d3ac2ff949dfb67"
    assert (
        namespace.cache_key("image-content-id")
        == "4abb0632bc2809625e5a16f8af1bb61f6209223da3303c4dca76defe0e46f119"
    )


def test_residency_event_wire_shape_matches_rust_contract() -> None:
    event = MMCacheResidencyEvent(
        worker_id=7,
        worker_generation=3,
        worker_role=WorkerRole.ENCODE,
        namespace=EmbeddingNamespace("example/model", "revision-1", "fingerprint"),
        cache_key="cache-key",
        action=ResidencyAction.ADD,
        bytes=4096,
        observed_at=1.234,
    )
    wire = to_wire(event)
    wire["observed_at_ms"] = round(wire.pop("observed_at") * 1000)
    assert canonical_wire_json(wire) == canonical_wire_json(
        {
            "worker_id": 7,
            "worker_generation": 3,
            "worker_role": "E",
            "namespace": {
                "model_id": "example/model",
                "model_revision": "revision-1",
                "processor_fingerprint": "fingerprint",
            },
            "cache_key": "cache-key",
            "action": "ADD",
            "bytes": 4096,
            "observed_at_ms": 1234,
        }
    )


def test_residency_index_ignores_stale_worker_generation() -> None:
    namespace = _namespace()
    obj = _object(namespace, 0)
    index = MMCacheResidencyIndex(ttl_seconds=30)
    _add(index, namespace, obj, worker_id=7, generation=2, role=WorkerRole.ENCODE)
    accepted = index.apply(
        MMCacheResidencyEvent(
            worker_id=7,
            worker_generation=1,
            worker_role=WorkerRole.ENCODE,
            namespace=namespace,
            cache_key=obj.embedding_cache_key,
            action=ResidencyAction.REMOVE,
            observed_at=11.0,
        )
    )
    assert not accepted
    assert (
        index.lookup(namespace, obj.embedding_cache_key, now=12.0)[0].worker_generation
        == 2
    )


def test_new_generation_clears_old_worker_entries() -> None:
    namespace = _namespace()
    objects = [_object(namespace, 0), _object(namespace, 1)]
    index = MMCacheResidencyIndex(ttl_seconds=30)
    for obj in objects:
        _add(index, namespace, obj, worker_id=7, generation=1, role=WorkerRole.ENCODE)
    _add(
        index,
        namespace,
        objects[1],
        worker_id=7,
        generation=2,
        role=WorkerRole.ENCODE,
        observed_at=11.0,
    )
    assert not index.lookup(namespace, objects[0].embedding_cache_key, now=12.0)
    assert (
        index.lookup(namespace, objects[1].embedding_cache_key, now=12.0)[
            0
        ].worker_generation
        == 2
    )


def test_planner_prefers_p_local_then_per_object_remote_holders() -> None:
    namespace = _namespace()
    objects = [_object(namespace, idx) for idx in range(3)]
    index = MMCacheResidencyIndex(ttl_seconds=30)
    _add(
        index, namespace, objects[0], worker_id=1, generation=3, role=WorkerRole.PREFILL
    )
    _add(
        index, namespace, objects[1], worker_id=11, generation=4, role=WorkerRole.ENCODE
    )
    _add(
        index, namespace, objects[2], worker_id=12, generation=5, role=WorkerRole.ENCODE
    )
    plan = JointMMPlanner(index).plan(
        namespace=namespace,
        objects=objects,
        prefill_candidates=[PrefillCandidate(1, 3, 5.0, 0.1)],
        encode_candidates=[
            EncodeCandidate(11, 4, 0.1, {}),
            EncodeCandidate(12, 5, 0.1, {}),
        ],
        now=12.0,
    )
    assert [item.source_kind for item in plan.objects] == [
        MMSourceKind.P_LOCAL,
        MMSourceKind.E_CACHE,
        MMSourceKind.E_CACHE,
    ]
    assert [item.source_worker_id for item in plan.objects] == [1, 11, 12]


def test_planner_jointly_scores_kv_ec_and_queue_delay() -> None:
    namespace = _namespace()
    obj = _object(namespace, 0)
    index = MMCacheResidencyIndex(ttl_seconds=30)
    _add(index, namespace, obj, worker_id=2, generation=1, role=WorkerRole.PREFILL)
    plan = JointMMPlanner(index).plan(
        namespace=namespace,
        objects=[obj],
        prefill_candidates=[
            PrefillCandidate(
                1, 1, predicted_saved_prefill_ms=10.0, queue_delay_ms=20.0
            ),
            PrefillCandidate(2, 1, predicted_saved_prefill_ms=4.0, queue_delay_ms=0.2),
        ],
        encode_candidates=[EncodeCandidate(9, 1, 0.0, {})],
        now=12.0,
    )
    assert plan.target_p_worker_id == 2
    assert plan.objects[0].source_kind is MMSourceKind.P_LOCAL


def test_uuid_only_object_fails_when_all_advertised_copies_are_gone() -> None:
    namespace = _namespace()
    obj = _object(namespace, 0, source_kind=ObjectSourceKind.UUID)
    with pytest.raises(ValueError, match="UUID-only object"):
        JointMMPlanner(MMCacheResidencyIndex()).plan(
            namespace=namespace,
            objects=[obj],
            prefill_candidates=[PrefillCandidate(1, 1, 0.0, 0.0)],
            encode_candidates=[EncodeCandidate(9, 1, 0.0, {})],
        )


def test_object_indices_must_preserve_request_order() -> None:
    namespace = _namespace()
    with pytest.raises(ValueError, match="contiguous indices"):
        JointMMPlanner(MMCacheResidencyIndex()).plan(
            namespace=namespace,
            objects=[_object(namespace, 1)],
            prefill_candidates=[PrefillCandidate(1, 1, 0.0, 0.0)],
            encode_candidates=[EncodeCandidate(9, 1, 0.0, {})],
        )
