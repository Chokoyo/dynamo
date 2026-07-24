# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang object-level multimodal EPD routing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from blake3 import blake3

from dynamo.common.multimodal_epd import MMObjectPlan, MMSourceKind
from dynamo.sglang.protocol import PreprocessedRequest, SglangEpdMediaObject

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"


@dataclass(frozen=True)
class PlannedMediaObject:
    object_index: int
    modality: Literal["IMAGE", "VIDEO"]
    url: str | None
    cache_key: str
    plan: MMObjectPlan


def sglang_image_cache_key(url: str) -> str:
    return blake3(url.encode()).hexdigest()


def extract_media_objects(request: PreprocessedRequest) -> list[SglangEpdMediaObject]:
    mm_data = request.multi_modal_data or {}
    objects: list[SglangEpdMediaObject] = []
    for modality, field_name in (("IMAGE", IMAGE_URL_KEY), ("VIDEO", VIDEO_URL_KEY)):
        for item in mm_data.get(field_name, []):
            if isinstance(item, str):
                url = item
                cache_key = sglang_image_cache_key(url) if modality == "IMAGE" else None
            elif isinstance(item, Mapping) and isinstance(item.get("Url"), str):
                url = item["Url"]
                cache_key = sglang_image_cache_key(url) if modality == "IMAGE" else None
            elif isinstance(item, Mapping) and isinstance(item.get("UuidOnly"), str):
                url = None
                cache_key = item["UuidOnly"]
            else:
                raise ValueError(f"unsupported {field_name} object for EPD: {item!r}")
            objects.append(
                SglangEpdMediaObject(
                    object_index=len(objects),
                    modality=modality,
                    url=url,
                    expected_cache_key=cache_key,
                )
            )
    return objects


def enforced_routing_plan(request: PreprocessedRequest) -> Mapping[str, Any] | None:
    routing_info = request.mm_routing_info
    if not isinstance(routing_info, Mapping):
        return None
    selection = routing_info.get("epd_prefill_selection")
    if not isinstance(selection, Mapping) or selection.get("mode") != "enforce":
        return None
    plan = routing_info.get("epd_routing_plan")
    return plan if isinstance(plan, Mapping) else None


def target_prefill_worker(request: PreprocessedRequest) -> int | None:
    routing_info = request.mm_routing_info
    if not isinstance(routing_info, Mapping):
        return None
    selection = routing_info.get("epd_prefill_selection")
    if not isinstance(selection, Mapping) or selection.get("mode") != "enforce":
        return None
    try:
        return int(selection["worker_id"])
    except (KeyError, TypeError, ValueError):
        return None


def parse_planned_media_objects(
    request: PreprocessedRequest,
    *,
    available_encode_worker_ids: set[int],
) -> tuple[PlannedMediaObject, ...] | None:
    routing_plan = enforced_routing_plan(request)
    if routing_plan is None:
        return None
    try:
        media_objects = extract_media_objects(request)
        raw_objects = routing_plan["objects"]
        if not isinstance(raw_objects, list) or len(raw_objects) != len(media_objects):
            raise ValueError("object plan must cover every media object exactly once")

        planned: list[PlannedMediaObject] = []
        for expected_index, (media, raw) in enumerate(
            zip(media_objects, raw_objects, strict=True)
        ):
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
            if (
                source_kind is not MMSourceKind.P_LOCAL
                and not available_encode_worker_ids
            ):
                raise ValueError("no encode worker is available for remote objects")
            if media.expected_cache_key is None:
                raise ValueError(
                    "planned SGLang EPD currently supports URL images only"
                )
            if media.url is None and source_kind is MMSourceKind.E_COMPUTE:
                raise ValueError("UUID-only objects cannot use E_COMPUTE")
            planned.append(
                PlannedMediaObject(
                    object_index=object_index,
                    modality=media.modality,
                    url=media.url,
                    cache_key=media.expected_cache_key,
                    plan=MMObjectPlan(
                        object_index=object_index,
                        source_kind=source_kind,
                        source_worker_id=worker_id,
                        source_worker_generation=worker_generation,
                        estimated_cost_ms=float(raw.get("estimated_cost_ms", 0.0)),
                    ),
                )
            )
        return tuple(planned)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def validate_object_response_indices(
    expected: Sequence[PlannedMediaObject], returned_indices: Sequence[int]
) -> None:
    expected_indices = [item.object_index for item in expected]
    if len(returned_indices) != len(set(returned_indices)):
        raise ValueError("encode response contains duplicate object indices")
    if sorted(returned_indices) != sorted(expected_indices):
        raise ValueError("encode response omitted or added multimodal objects")
