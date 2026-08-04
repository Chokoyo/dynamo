# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

import torch
from vllm.sampling_params import SamplingParams as VllmSamplingParams

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal_epd import MMObjectPlan, MMSourceKind
from dynamo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    LocalEmbeddingReceiver,
)
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.llm import MultimodalEmbeddingCachePublisher
from dynamo.runtime import Client

from .encode_utils import get_embedding_hash
from .model import construct_mm_data
from .protocol import (
    MultiModalGroup,
    MultiModalInput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

SPLIT_ENCODE = int(os.getenv("DYN_SPLIT_ENCODE", 1))


# ── Internal helpers (all underscore-prefixed) ───────────────────────


class _PendingRelease:
    """Tracks NIXL tensor buffers that should be released after consumption.

    For NIXL receivers, embeddings are views into pre-allocated reusable
    buffers.  Instead of cloning each embedding eagerly, we defer the
    release until the caller has consumed the tensors (e.g. via
    ``_accumulate_embeddings`` which copies data through ``torch.cat``).
    """

    __slots__ = ("_receiver", "_tensor_ids")

    def __init__(self, receiver: AbstractEmbeddingReceiver):
        self._receiver = receiver
        self._tensor_ids: List[int] = []

    def track(self, tensor_id: int) -> None:
        self._tensor_ids.append(tensor_id)

    def release_all(self) -> None:
        for tid in self._tensor_ids:
            self._receiver.release_tensor(tid)
        self._tensor_ids.clear()

    def merge(self, other: "_PendingRelease | None") -> None:
        if other is None:
            return
        if other._receiver is not self._receiver:
            raise ValueError("cannot merge pending releases from different receivers")
        self._tensor_ids.extend(other._tensor_ids)
        other._tensor_ids.clear()


def _parse_object_plans(
    routing_plan: Mapping[str, Any] | None,
    *,
    object_count: int,
    available_encode_worker_ids: set[int],
    available_prefill_worker_ids: set[int],
) -> tuple[MMObjectPlan, ...] | None:
    """Validate an enforced object plan or return ``None`` for legacy routing."""
    if routing_plan is None:
        return None
    try:
        raw_objects = routing_plan["objects"]
        if not isinstance(raw_objects, list) or len(raw_objects) != object_count:
            raise ValueError("object plan must cover every image exactly once")

        plans: list[MMObjectPlan] = []
        for expected_index, raw in enumerate(raw_objects):
            if not isinstance(raw, Mapping):
                raise ValueError("object plan entries must be objects")
            object_index = int(raw["object_index"])
            if object_index != expected_index:
                raise ValueError("object plan indices must be contiguous and ordered")
            source_kind = MMSourceKind(str(raw["source_kind"]))
            worker_id = int(raw["source_worker_id"])
            worker_generation = int(raw["source_worker_generation"])
            if worker_generation < 0:
                raise ValueError("worker generation must be non-negative")
            if source_kind is MMSourceKind.P_REMOTE:
                if worker_id not in available_prefill_worker_ids:
                    raise ValueError(
                        f"planned prefill worker {worker_id} is unavailable"
                    )
            elif (
                source_kind is not MMSourceKind.P_LOCAL
                and worker_id not in available_encode_worker_ids
            ):
                raise ValueError(f"planned encode worker {worker_id} is unavailable")
            plans.append(
                MMObjectPlan(
                    object_index=object_index,
                    source_kind=source_kind,
                    source_worker_id=worker_id,
                    source_worker_generation=worker_generation,
                    estimated_cost_ms=float(raw.get("estimated_cost_ms", 0.0)),
                )
            )
        return tuple(plans)
    except (KeyError, TypeError, ValueError) as error:
        logger.warning("Ignoring invalid multimodal EPD object plan: %s", error)
        return None


def _accumulate_embeddings(
    multi_modal_data: Dict[str, Any],
    model: str,
    embeddings_dtype: torch.dtype,
    embeddings: torch.Tensor,
    image_grid_thw,
) -> None:
    """Construct model-specific mm_data from embeddings and merge into the
    accumulated ``multi_modal_data`` dict (mutated in-place).

    Handles both video (numpy conversion) and image modalities, including
    the Qwen-VL dict-style embeddings with ``image_embeds`` + ``image_grid_thw``.
    """
    if "video" in model.lower():
        video_numpy = embeddings.numpy()
        mm_data = construct_mm_data(
            model,
            embeddings_dtype,
            video_numpy=video_numpy,
        )
        multi_modal_data["video"].append(mm_data["video"])
        return

    mm_data = construct_mm_data(
        model,
        embeddings_dtype,
        image_embeds=embeddings,
        image_grid_thw=image_grid_thw,
    )

    if "image" not in multi_modal_data:
        multi_modal_data["image"] = mm_data["image"]
        return

    if isinstance(mm_data["image"], dict):
        # Qwen-VL style: dict with image_embeds + image_grid_thw tensors
        multi_modal_data["image"]["image_embeds"] = torch.cat(
            (
                multi_modal_data["image"]["image_embeds"],
                mm_data["image"]["image_embeds"],
            )
        )
        multi_modal_data["image"]["image_grid_thw"] = torch.cat(
            (
                multi_modal_data["image"]["image_grid_thw"],
                mm_data["image"]["image_grid_thw"],
            )
        )
    elif isinstance(mm_data["image"], torch.Tensor):
        multi_modal_data["image"] = torch.cat(
            (multi_modal_data["image"], mm_data["image"])
        )
    else:
        raise ValueError(
            f"Unexpected image data format from construct_mm_data: {type(mm_data['image'])}"
        )


def _ensure_owned_tensors(multi_modal_data: Dict[str, Any]) -> None:
    """Clone tensor views so NIXL buffers can be safely released.

    Only needed for single-image; multi-image goes through torch.cat
    which already produces owned tensors.
    """
    img = multi_modal_data.get("image")
    if isinstance(img, dict):
        for k, v in img.items():
            if isinstance(v, torch.Tensor):
                img[k] = v.clone()
    elif isinstance(img, torch.Tensor):
        multi_modal_data["image"] = img.clone()


async def _fetch_from_encode_workers(
    encode_worker_client: Client,
    image_urls: List[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    object_plans: Sequence[MMObjectPlan] | None = None,
    context=None,
    worker_label: str = "encode",
    retry_alternate_worker: bool = True,
) -> tuple[List[MultiModalGroup], _PendingRelease | None]:
    """Fan out image URLs to encode workers, load embeddings, and return ready groups.

    For NIXL receivers the returned embeddings are zero-copy views into
    pre-allocated buffers.  The returned ``_PendingRelease`` must be
    released after the tensors have been consumed.
    """
    encode_worker_ids = encode_worker_client.instance_ids()
    encode_worker_count = len(encode_worker_ids)
    if encode_worker_count == 0:
        raise RuntimeError(f"No {worker_label} workers available for multimodal input")
    if object_plans is not None and len(object_plans) != len(image_urls):
        raise ValueError("object plan count must match the number of image URLs")

    encode_batch_size = (
        max(1, len(image_urls) // encode_worker_count)
        if SPLIT_ENCODE
        else len(image_urls)
    )

    def make_request(groups: list[MultiModalGroup]) -> str:
        return vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(prompt_token_ids=[]),
            sampling_params=VllmSamplingParams(),
            request_id=request_id,
            multimodal_inputs=groups,
        ).model_dump_json()

    async def collect(stream) -> list[MultiModalGroup]:
        collected: list[MultiModalGroup] = []
        async for response in stream:
            output = vLLMMultimodalRequest.model_validate_json(response.data())  # type: ignore[attr-defined]
            if output.multimodal_inputs:
                collected.extend(output.multimodal_inputs)
        return collected

    def validate_response(indices: list[int], collected: list[MultiModalGroup]) -> None:
        if len(indices) != len(collected):
            raise ValueError(
                "encode worker returned a different object count than requested"
            )
        for index, group in zip(indices, collected, strict=True):
            returned_input = group.multimodal_input
            returned_url = (
                returned_input.image_url if returned_input is not None else None
            )
            if returned_url != image_urls[index]:
                raise ValueError(
                    "encode worker returned a multimodal object with a mismatched URL"
                )

    async def dispatch_and_collect(
        indices: list[int], worker_id: int | None
    ) -> list[MultiModalGroup]:
        payload = make_request([groups[index] for index in indices])
        if worker_id is None:
            stream = await encode_worker_client.round_robin(  # type: ignore[arg-type]
                payload, context=context
            )
        else:
            stream = await encode_worker_client.direct(  # type: ignore[arg-type]
                payload,
                worker_id,
                context=context,
            )
        collected = await collect(stream)
        validate_response(indices, collected)
        return collected

    def alternate_worker(failed_worker_id: int) -> int | None:
        return next(
            (
                worker_id
                for worker_id in encode_worker_ids
                if worker_id != failed_worker_id
            ),
            None,
        )

    with time_and_log_code_section(f"[PREFILL] request: {request_id} dispatch encode"):
        groups = [
            MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
            for url in image_urls
        ]
        dispatches: list[tuple[list[int], int | None]] = []
        if object_plans is None:
            for start in range(0, len(groups), encode_batch_size):
                indices = list(
                    range(start, min(start + encode_batch_size, len(groups)))
                )
                dispatches.append((indices, None))
        else:
            by_worker: dict[int, list[int]] = defaultdict(list)
            for index, plan in enumerate(object_plans):
                if plan.source_kind is MMSourceKind.P_LOCAL:
                    raise ValueError(
                        f"P_LOCAL object cannot be dispatched to a {worker_label} worker"
                    )
                by_worker[plan.source_worker_id].append(index)
            dispatches = [
                (indices, worker_id) for worker_id, indices in by_worker.items()
            ]

    with time_and_log_code_section(
        f"[PREFILL] request: {request_id} receive encode responses"
    ):
        collected_batches = await asyncio.gather(
            *(
                dispatch_and_collect(indices, worker_id)
                for indices, worker_id in dispatches
            ),
            return_exceptions=True,
        )
        ordered_groups: list[MultiModalGroup | None] = [None] * len(image_urls)
        for (indices, worker_id), collected in zip(
            dispatches, collected_batches, strict=True
        ):
            if isinstance(collected, BaseException):
                if worker_id is None:
                    raise collected
                if not retry_alternate_worker:
                    raise collected
                retry_worker_id = alternate_worker(worker_id)
                logger.warning(
                    "Planned multimodal %s worker %s failed validation for request "
                    "%s; retrying %s object(s) via %s",
                    worker_label,
                    worker_id,
                    request_id,
                    len(indices),
                    (
                        f"{worker_label} worker {retry_worker_id}"
                        if retry_worker_id is not None
                        else "legacy round-robin routing"
                    ),
                    exc_info=(
                        type(collected),
                        collected,
                        collected.__traceback__,
                    ),
                )
                collected = await dispatch_and_collect(indices, retry_worker_id)
            for index, group in zip(indices, collected, strict=True):
                ordered_groups[index] = group
        if any(group is None for group in ordered_groups):
            raise ValueError("encode worker response omitted a multimodal object")
        multimodal_groups = [group for group in ordered_groups if group is not None]

    with time_and_log_code_section(
        f"[PREFILL] request: {request_id} receive embeddings"
    ):
        tasks = [
            asyncio.create_task(receiver.receive_embeddings(group.serialized_request))
            for group in multimodal_groups
            if group.serialized_request is not None
        ]
        loaded = await asyncio.gather(*tasks)

    is_local = isinstance(receiver, LocalEmbeddingReceiver)
    pending: _PendingRelease | None = None if is_local else _PendingRelease(receiver)
    for group, (tensor_id, embedding) in zip(multimodal_groups, loaded, strict=True):
        group.loaded_embedding = embedding
        if pending is not None:
            pending.track(tensor_id)

    return multimodal_groups, pending


async def _fetch_from_prefill_workers(
    prefill_worker_client: Client,
    image_urls: List[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    object_plans: Sequence[MMObjectPlan],
    context=None,
) -> tuple[List[MultiModalGroup], _PendingRelease | None]:
    if any(plan.source_kind is not MMSourceKind.P_REMOTE for plan in object_plans):
        raise ValueError("prefill fetch only accepts P_REMOTE object plans")
    return await _fetch_from_encode_workers(
        prefill_worker_client,
        image_urls,
        request_id,
        receiver,
        object_plans=object_plans,
        context=context,
        worker_label="prefill embedding-fetch",
        retry_alternate_worker=False,
    )


async def _fetch_embeddings(
    encode_worker_client: Client,
    image_urls: list[str],
    request_id: str,
    receiver: AbstractEmbeddingReceiver,
    cache: MultimodalEmbeddingCacheManager | None = None,
    cache_publisher: MultimodalEmbeddingCachePublisher | None = None,
    routing_plan: Mapping[str, Any] | None = None,
    prefill_worker_client: Client | None = None,
    context=None,
) -> tuple[list[MultiModalGroup], _PendingRelease | None]:
    """Fetch multimodal embeddings with transparent cache-through.

    Pipeline: check_cache → fetch misses from encode workers → update_cache.
    When *cache* is ``None`` the cache steps are no-ops and all URLs go
    straight to the encode workers.

    For NIXL receivers the returned embeddings are zero-copy views.  The
    returned ``_PendingRelease`` must be released after consuming the
    tensors.
    """
    object_plans = (
        _parse_object_plans(
            routing_plan,
            object_count=len(image_urls),
            available_encode_worker_ids=set(encode_worker_client.instance_ids()),
            available_prefill_worker_ids=(
                set(prefill_worker_client.instance_ids())
                if prefill_worker_client is not None
                else set()
            ),
        )
        if routing_plan is not None
        else None
    )
    results: list[MultiModalGroup | None] = [None] * len(image_urls)
    to_fetch: list[tuple[int, str, str | None, MMObjectPlan | None]] = []

    # ── 1. Check cache (no-op when cache is None) ────────────────────
    for idx, url in enumerate(image_urls):
        if cache is not None:
            key = get_embedding_hash(url)
            cached = cache.get(key)
            if cached is not None:
                logger.debug(f"[{request_id}] Cache hit for URL index {idx}")
                results[idx] = MultiModalGroup(
                    loaded_embedding=cached.tensor,
                    image_grid_thw=cached.image_grid_thw,
                )
                continue
        else:
            key = None
        object_plan = object_plans[idx] if object_plans is not None else None
        if object_plan is not None and object_plan.source_kind is MMSourceKind.P_LOCAL:
            logger.warning(
                "Planned P-local embedding is absent for request %s object %s; "
                "falling back to encode routing",
                request_id,
                idx,
            )
            object_plan = None
        to_fetch.append((idx, url, key, object_plan))

    # ── 2. Fetch uncached from planned P/E sources ───────────────────
    pending: _PendingRelease | None = None
    if to_fetch:
        p_remote = [
            item
            for item in to_fetch
            if item[3] is not None and item[3].source_kind is MMSourceKind.P_REMOTE
        ]
        planned_encode = [
            item
            for item in to_fetch
            if item[3] is not None
            and item[3].source_kind in (MMSourceKind.E_CACHE, MMSourceKind.E_COMPUTE)
        ]
        legacy_encode = [item for item in to_fetch if item[3] is None]

        async def store_groups(
            entries: list[tuple[int, str, str | None, MMObjectPlan | None]],
            groups: list[MultiModalGroup],
            fetched_pending: _PendingRelease | None,
        ) -> None:
            nonlocal pending
            if pending is None:
                pending = fetched_pending
            elif fetched_pending is not None:
                pending.merge(fetched_pending)
            for (idx, _url, key, _plan), group in zip(entries, groups, strict=True):
                if cache is not None and key is not None:
                    assert group.loaded_embedding is not None
                    mutation = cache.set_with_delta(
                        key,
                        CachedEmbedding(
                            tensor=group.loaded_embedding.clone(),
                            image_grid_thw=group.image_grid_thw,
                        ),
                    )
                    if cache_publisher is not None:
                        try:
                            cache_publisher.publish_delta(
                                mutation.added_keys, mutation.removed_keys
                            )
                        except Exception:
                            logger.warning(
                                "Failed to publish vLLM prefill cache delta",
                                exc_info=True,
                            )
                results[idx] = group

        if p_remote:
            by_prefill_worker: dict[
                int, list[tuple[int, str, str | None, MMObjectPlan | None]]
            ] = defaultdict(list)
            for entry in p_remote:
                plan = entry[3]
                assert plan is not None
                by_prefill_worker[plan.source_worker_id].append(entry)

            for source_worker_id, entries in by_prefill_worker.items():
                try:
                    if prefill_worker_client is None:
                        raise RuntimeError(
                            "prefill embedding-fetch client is unavailable"
                        )
                    groups, fetched_pending = await _fetch_from_prefill_workers(
                        prefill_worker_client,
                        [url for _, url, _, _ in entries],
                        request_id,
                        receiver,
                        [plan for _, _, _, plan in entries if plan is not None],
                        context=context,
                    )
                except Exception:
                    logger.warning(
                        "Remote prefill embedding fetch from worker %s failed for "
                        "request %s; falling back to encode routing for %s object(s)",
                        source_worker_id,
                        request_id,
                        len(entries),
                        exc_info=True,
                    )
                    legacy_encode.extend(
                        [(idx, url, key, None) for idx, url, key, _ in entries]
                    )
                else:
                    await store_groups(entries, groups, fetched_pending)

        if planned_encode:
            groups, fetched_pending = await _fetch_from_encode_workers(
                encode_worker_client,
                [url for _, url, _, _ in planned_encode],
                request_id,
                receiver,
                object_plans=[
                    plan for _, _, _, plan in planned_encode if plan is not None
                ],
                context=context,
            )
            await store_groups(planned_encode, groups, fetched_pending)

        if legacy_encode:
            groups, fetched_pending = await _fetch_from_encode_workers(
                encode_worker_client,
                [url for _, url, _, _ in legacy_encode],
                request_id,
                receiver,
                object_plans=None,
                context=context,
            )
            await store_groups(legacy_encode, groups, fetched_pending)

    return [r for r in results if r is not None], pending


# ── Public API (single entry point) ─────────────────────────────────


class MultiModalEmbeddingLoader:
    """Helper class for requesting remote encode and receive embeddings."""

    def __init__(
        self,
        encode_worker_client: Client,
        receiver: AbstractEmbeddingReceiver,
        embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None,
        cache_publisher: MultimodalEmbeddingCachePublisher | None = None,
        prefill_worker_client: Client | None = None,
    ):
        self._encode_worker_client = encode_worker_client
        self._prefill_worker_client = prefill_worker_client
        self._receiver = receiver
        self._embedding_cache_manager = embedding_cache_manager
        self._cache_publisher = cache_publisher

    async def load_multimodal_embedding_parts(
        self,
        image_urls: list[str],
        request_id: str,
        *,
        model: str,
        routing_plan: Mapping[str, Any] | None = None,
        context=None,
    ) -> list[CachedEmbedding]:
        """Fetch ordered, owned embedding parts for the post-KV connector path."""
        if self._encode_worker_client is None or not image_urls:
            return []
        groups, pending = await _fetch_embeddings(
            self._encode_worker_client,
            image_urls,
            request_id,
            self._receiver,
            cache=self._embedding_cache_manager,
            cache_publisher=self._cache_publisher,
            routing_plan=routing_plan,
            prefill_worker_client=self._prefill_worker_client,
            context=context,
        )
        parts = []
        for group in groups:
            if group.loaded_embedding is None:
                raise ValueError("encode worker returned no embedding tensor")
            parts.append(
                CachedEmbedding(
                    tensor=group.loaded_embedding.clone(),
                    image_grid_thw=group.image_grid_thw,
                )
            )
        if pending is not None:
            pending.release_all()
        return parts

    async def load_multimodal_embeddings(
        self,
        image_urls: list[str],
        request_id: str,
        *,
        model: str,
        routing_plan: Mapping[str, Any] | None = None,
        context=None,
    ) -> Dict[str, Any]:
        """Fetch embeddings and build engine-ready ``multi_modal_data``.

        Full pipeline:
        cache check → remote fetch → cache update → accumulate → release NIXL buffers.

        Returns a dict suitable for passing to ``TokensPrompt(multi_modal_data=...)``.
        """
        if self._encode_worker_client is None or not image_urls:
            return {}

        groups, pending = await _fetch_embeddings(
            self._encode_worker_client,
            image_urls,
            request_id,
            self._receiver,
            cache=self._embedding_cache_manager,
            cache_publisher=self._cache_publisher,
            routing_plan=routing_plan,
            prefill_worker_client=self._prefill_worker_client,
            context=context,
        )

        multi_modal_data: Dict[str, Any] = {}
        with time_and_log_code_section(
            f"[PREFILL] request: {request_id} accumulate embeddings"
        ):
            for group in groups:
                assert group.loaded_embedding is not None
                _accumulate_embeddings(
                    multi_modal_data,
                    model,
                    group.loaded_embedding.dtype,
                    group.loaded_embedding,
                    group.image_grid_thw,
                )

        if pending is not None:
            # Multi-image: torch.cat in _accumulate_embeddings already created
            # owned tensors.  Single-image: the data is still a view into the
            # NIXL buffer, so we must clone before releasing.
            if len(groups) == 1:
                _ensure_owned_tensors(multi_modal_data)
            pending.release_all()

        return multi_modal_data

    def get_cached_embedding(self, cache_key: str) -> CachedEmbedding | None:
        if self._embedding_cache_manager is None:
            return None
        return self._embedding_cache_manager.get(cache_key)
