# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight contracts and planning primitives for cache-aware multimodal EPD."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class WorkerRole(str, Enum):
    ENCODE = "E"
    PREFILL = "P"


class ResidencyAction(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    CLEAR = "CLEAR"


class MMSourceKind(str, Enum):
    P_LOCAL = "P_LOCAL"
    P_REMOTE = "P_REMOTE"
    E_CACHE = "E_CACHE"
    E_COMPUTE = "E_COMPUTE"


class RoutingMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    ENFORCE = "enforce"


class ObjectSourceKind(str, Enum):
    URL = "URL"
    DECODED = "DECODED"
    INLINE = "INLINE"
    UUID = "UUID"


def to_wire(value: Any) -> Any:
    """Convert a shared EPD contract to its JSON-compatible wire shape."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_wire(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_wire(item) for item in value]
    return value


def canonical_wire_json(value: Any) -> str:
    return json.dumps(
        to_wire(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


@dataclass(frozen=True, slots=True)
class EmbeddingNamespace:
    model_id: str
    model_revision: str
    processor_fingerprint: str

    @classmethod
    def from_processor_config(
        cls,
        *,
        model_id: str,
        model_revision: str,
        processor_config: Mapping[str, Any],
    ) -> "EmbeddingNamespace":
        canonical = json.dumps(
            processor_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        fingerprint = hashlib.blake2b(canonical, digest_size=16).hexdigest()
        return cls(model_id, model_revision, fingerprint)

    def cache_key(self, content_id: str) -> str:
        payload = "\0".join(
            (
                self.model_id,
                self.model_revision,
                self.processor_fingerprint,
                content_id,
            )
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=32).hexdigest()


@dataclass(frozen=True, slots=True)
class MMTokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid multimodal token span: {self.start}:{self.end}")


@dataclass(frozen=True, slots=True)
class MMObjectRef:
    object_index: int
    modality: str
    content_id: str
    embedding_cache_key: str
    model_token_spans: tuple[MMTokenSpan, ...]
    source_kind: ObjectSourceKind
    model_visible_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.object_index < 0:
            raise ValueError("object_index must be non-negative")
        if not self.content_id or not self.embedding_cache_key:
            raise ValueError("content_id and embedding_cache_key are required")
        if not self.model_token_spans:
            raise ValueError("at least one model token span is required")


@dataclass(frozen=True, slots=True)
class CacheLocation:
    worker_id: int
    worker_generation: int
    worker_role: WorkerRole
    bytes: int
    observed_at: float


@dataclass(frozen=True, slots=True)
class MMCacheResidencyEvent:
    worker_id: int
    worker_generation: int
    worker_role: WorkerRole
    namespace: EmbeddingNamespace
    cache_key: str | None
    action: ResidencyAction
    bytes: int = 0
    observed_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.worker_generation < 0:
            raise ValueError("worker_generation must be non-negative")
        if self.action is not ResidencyAction.CLEAR and not self.cache_key:
            raise ValueError(f"cache_key is required for {self.action.value}")
        if self.bytes < 0:
            raise ValueError("bytes must be non-negative")


class MMCacheResidencyIndex:
    """Role-aware, generation-safe index of embedding-cache ownership."""

    def __init__(self, *, ttl_seconds: float = 30.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._worker_generations: dict[tuple[WorkerRole, int], int] = {}
        self._entries: dict[
            tuple[EmbeddingNamespace, str], dict[tuple[WorkerRole, int], CacheLocation]
        ] = {}

    def apply(self, event: MMCacheResidencyEvent) -> bool:
        worker_key = (event.worker_role, event.worker_id)
        current_generation = self._worker_generations.get(worker_key)
        if (
            current_generation is not None
            and event.worker_generation < current_generation
        ):
            return False
        if current_generation is None or event.worker_generation > current_generation:
            self._clear_worker(event.worker_role, event.worker_id)
            self._worker_generations[worker_key] = event.worker_generation

        if event.action is ResidencyAction.CLEAR:
            self._clear_worker(event.worker_role, event.worker_id)
            return True

        entry_key = (event.namespace, event.cache_key or "")
        locations = self._entries.setdefault(entry_key, {})
        if event.action is ResidencyAction.REMOVE:
            locations.pop(worker_key, None)
            if not locations:
                self._entries.pop(entry_key, None)
            return True

        locations[worker_key] = CacheLocation(
            worker_id=event.worker_id,
            worker_generation=event.worker_generation,
            worker_role=event.worker_role,
            bytes=event.bytes,
            observed_at=event.observed_at,
        )
        return True

    def replace_worker_snapshot(
        self,
        *,
        worker_id: int,
        worker_generation: int,
        worker_role: WorkerRole,
        namespace: EmbeddingNamespace,
        entries: Mapping[str, int],
        observed_at: float | None = None,
    ) -> None:
        timestamp = time.monotonic() if observed_at is None else observed_at
        self.apply(
            MMCacheResidencyEvent(
                worker_id=worker_id,
                worker_generation=worker_generation,
                worker_role=worker_role,
                namespace=namespace,
                cache_key=None,
                action=ResidencyAction.CLEAR,
                observed_at=timestamp,
            )
        )
        for cache_key, size_bytes in entries.items():
            self.apply(
                MMCacheResidencyEvent(
                    worker_id=worker_id,
                    worker_generation=worker_generation,
                    worker_role=worker_role,
                    namespace=namespace,
                    cache_key=cache_key,
                    action=ResidencyAction.ADD,
                    bytes=size_bytes,
                    observed_at=timestamp,
                )
            )

    def lookup(
        self,
        namespace: EmbeddingNamespace,
        cache_key: str,
        *,
        role: WorkerRole | None = None,
        now: float | None = None,
    ) -> tuple[CacheLocation, ...]:
        timestamp = time.monotonic() if now is None else now
        locations = self._entries.get((namespace, cache_key), {})
        live = [
            location
            for location in locations.values()
            if timestamp - location.observed_at <= self._ttl_seconds
            and (role is None or location.worker_role is role)
        ]
        live.sort(key=lambda item: (item.worker_role.value, item.worker_id))
        return tuple(live)

    def prune(self, *, now: float | None = None) -> int:
        timestamp = time.monotonic() if now is None else now
        removed = 0
        for entry_key, locations in list(self._entries.items()):
            for worker_key, location in list(locations.items()):
                if timestamp - location.observed_at > self._ttl_seconds:
                    locations.pop(worker_key)
                    removed += 1
            if not locations:
                self._entries.pop(entry_key)
        return removed

    def _clear_worker(self, role: WorkerRole, worker_id: int) -> None:
        worker_key = (role, worker_id)
        for entry_key, locations in list(self._entries.items()):
            locations.pop(worker_key, None)
            if not locations:
                self._entries.pop(entry_key)


@dataclass(frozen=True, slots=True)
class PrefillCandidate:
    worker_id: int
    worker_generation: int
    predicted_saved_prefill_ms: float
    queue_delay_ms: float
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class EncodeCandidate:
    worker_id: int
    worker_generation: int
    queue_delay_ms: float
    encode_ms_by_key: Mapping[str, float]
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class MMObjectPlan:
    object_index: int
    source_kind: MMSourceKind
    source_worker_id: int
    source_worker_generation: int
    estimated_cost_ms: float


@dataclass(frozen=True, slots=True)
class MMRoutingPlan:
    target_p_worker_id: int
    target_p_generation: int
    objects: tuple[MMObjectPlan, ...]
    predicted_benefit_ms: float
    score_components: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class JointPlannerConfig:
    remote_bandwidth_bytes_per_ms: float = 100_000.0
    remote_transfer_fixed_ms: float = 0.05
    full_kv_recompute_threshold_ms: float = 5.0
    fanout_penalty_ms: float = 0.15
    local_ec_saved_encode_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.remote_bandwidth_bytes_per_ms <= 0:
            raise ValueError("remote_bandwidth_bytes_per_ms must be positive")


class JointMMPlanner:
    def __init__(
        self,
        residency_index: MMCacheResidencyIndex,
        config: JointPlannerConfig | None = None,
    ) -> None:
        self._residency_index = residency_index
        self._config = config or JointPlannerConfig()

    def plan(
        self,
        *,
        namespace: EmbeddingNamespace,
        objects: Sequence[MMObjectRef],
        prefill_candidates: Sequence[PrefillCandidate],
        encode_candidates: Sequence[EncodeCandidate],
        now: float | None = None,
    ) -> MMRoutingPlan:
        self._validate_objects(objects)
        healthy_p = [candidate for candidate in prefill_candidates if candidate.healthy]
        healthy_e = [candidate for candidate in encode_candidates if candidate.healthy]
        if not healthy_p:
            raise ValueError("no healthy prefill workers are available")
        timestamp = time.monotonic() if now is None else now
        locations_by_key = {
            obj.embedding_cache_key: self._residency_index.lookup(
                namespace,
                obj.embedding_cache_key,
                now=timestamp,
            )
            for obj in objects
        }
        encoders_by_worker = {
            (candidate.worker_id, candidate.worker_generation): candidate
            for candidate in healthy_e
        }
        prefills_by_worker = {
            (candidate.worker_id, candidate.worker_generation): candidate
            for candidate in healthy_p
        }

        plans = [
            self._plan_for_prefill(
                objects=objects,
                prefill=candidate,
                prefills_by_worker=prefills_by_worker,
                encoders_by_worker=encoders_by_worker,
                locations_by_key=locations_by_key,
            )
            for candidate in healthy_p
        ]
        plans.sort(
            key=lambda plan: (-plan.predicted_benefit_ms, plan.target_p_worker_id)
        )
        return plans[0]

    def _plan_for_prefill(
        self,
        *,
        objects: Sequence[MMObjectRef],
        prefill: PrefillCandidate,
        prefills_by_worker: Mapping[tuple[int, int], PrefillCandidate],
        encoders_by_worker: Mapping[tuple[int, int], EncodeCandidate],
        locations_by_key: Mapping[str, tuple[CacheLocation, ...]],
    ) -> MMRoutingPlan:
        object_plans: list[MMObjectPlan] = []
        selected_remote_workers: set[int] = set()
        encode_saved_ms = 0.0
        transfer_ms = 0.0
        encoder_queue_ms = 0.0
        encode_compute_ms = 0.0

        for obj in objects:
            locations = locations_by_key[obj.embedding_cache_key]
            local = next(
                (
                    location
                    for location in locations
                    if location.worker_role is WorkerRole.PREFILL
                    and location.worker_id == prefill.worker_id
                    and location.worker_generation == prefill.worker_generation
                ),
                None,
            )
            if local is not None:
                object_plans.append(
                    MMObjectPlan(
                        obj.object_index,
                        MMSourceKind.P_LOCAL,
                        local.worker_id,
                        local.worker_generation,
                        0.0,
                    )
                )
                encode_saved_ms += self._config.local_ec_saved_encode_ms
                continue

            remote_options: list[tuple[float, MMSourceKind, CacheLocation]] = []
            for location in locations:
                if location.worker_role is WorkerRole.PREFILL:
                    remote_prefill = prefills_by_worker.get(
                        (location.worker_id, location.worker_generation)
                    )
                    if (
                        remote_prefill is None
                        or location.worker_id == prefill.worker_id
                    ):
                        continue
                    remote_cost = (
                        self._config.remote_transfer_fixed_ms
                        + location.bytes / self._config.remote_bandwidth_bytes_per_ms
                    )
                    remote_options.append(
                        (remote_cost, MMSourceKind.P_REMOTE, location)
                    )
                elif location.worker_role is WorkerRole.ENCODE:
                    encoder = encoders_by_worker.get(
                        (location.worker_id, location.worker_generation)
                    )
                    if encoder is None:
                        continue
                    remote_cost = (
                        encoder.queue_delay_ms
                        + self._config.remote_transfer_fixed_ms
                        + location.bytes / self._config.remote_bandwidth_bytes_per_ms
                    )
                    remote_options.append((remote_cost, MMSourceKind.E_CACHE, location))

            if remote_options:
                remote_cost, source_kind, location = min(
                    remote_options, key=lambda item: (item[0], item[2].worker_id)
                )
                selected_remote_workers.add(location.worker_id)
                transfer_ms += remote_cost
                object_plans.append(
                    MMObjectPlan(
                        obj.object_index,
                        source_kind,
                        location.worker_id,
                        location.worker_generation,
                        remote_cost,
                    )
                )
                encode_saved_ms += self._config.local_ec_saved_encode_ms
                continue

            if obj.source_kind is ObjectSourceKind.UUID:
                raise ValueError(
                    f"UUID-only object {obj.object_index} has no live cache holder"
                )
            compute = self._select_compute_worker(
                obj, tuple(encoders_by_worker.values())
            )
            if compute is None:
                raise ValueError("no healthy encode workers are available")
            object_encode_ms = compute.encode_ms_by_key.get(
                obj.embedding_cache_key, self._config.local_ec_saved_encode_ms
            )
            encode_cost = compute.queue_delay_ms + object_encode_ms
            selected_remote_workers.add(compute.worker_id)
            encoder_queue_ms += compute.queue_delay_ms
            encode_compute_ms += object_encode_ms
            object_plans.append(
                MMObjectPlan(
                    obj.object_index,
                    MMSourceKind.E_COMPUTE,
                    compute.worker_id,
                    compute.worker_generation,
                    encode_cost,
                )
            )

        fanout_penalty = max(0, len(selected_remote_workers) - 1) * (
            self._config.fanout_penalty_ms
        )
        # TODO: Ablate each heuristic score component independently before choosing
        # production defaults. Calibrate the weights from measured TTFT and verify
        # which terms improve routing beyond KV overlap plus projected prefill cost.
        predicted_benefit = (
            prefill.predicted_saved_prefill_ms
            + encode_saved_ms
            - prefill.queue_delay_ms
            - transfer_ms
            - encoder_queue_ms
            - encode_compute_ms
            - fanout_penalty
        )
        if (
            prefill.predicted_saved_prefill_ms > 0
            and prefill.queue_delay_ms <= self._config.full_kv_recompute_threshold_ms
        ):
            predicted_benefit += self._config.full_kv_recompute_threshold_ms

        return MMRoutingPlan(
            target_p_worker_id=prefill.worker_id,
            target_p_generation=prefill.worker_generation,
            objects=tuple(sorted(object_plans, key=lambda item: item.object_index)),
            predicted_benefit_ms=predicted_benefit,
            score_components={
                "predicted_saved_prefill_ms": prefill.predicted_saved_prefill_ms,
                "local_or_remote_ec_saved_encode_ms": encode_saved_ms,
                "p_queue_delay_ms": prefill.queue_delay_ms,
                "remote_transfer_ms": transfer_ms,
                "encoder_queue_delay_ms": encoder_queue_ms,
                "encode_compute_ms": encode_compute_ms,
                "fanout_penalty_ms": fanout_penalty,
            },
        )

    @staticmethod
    def _select_compute_worker(
        obj: MMObjectRef,
        candidates: Sequence[EncodeCandidate],
    ) -> EncodeCandidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                candidate.queue_delay_ms
                + candidate.encode_ms_by_key.get(obj.embedding_cache_key, 1.0),
                candidate.worker_id,
            ),
        )

    @staticmethod
    def _validate_objects(objects: Iterable[MMObjectRef]) -> None:
        indices = [obj.object_index for obj in objects]
        if indices != list(range(len(indices))):
            raise ValueError(
                "multimodal objects must have unique contiguous indices in request order"
            )
