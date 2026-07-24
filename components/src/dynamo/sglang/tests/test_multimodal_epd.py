# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import TransferRequest
from dynamo.common.multimodal_epd import MMSourceKind
from dynamo.sglang.multimodal_epd import (
    parse_planned_media_objects,
    sglang_image_cache_key,
    target_prefill_worker,
    validate_object_response_indices,
)
from dynamo.sglang.protocol import (
    DisaggSglangMultimodalRequest,
    PreprocessedRequest,
    SamplingOptions,
    SglangEpdObjectPart,
    SglangEpdObjectRequest,
    SglangEpdObjectResponse,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    MultimodalEncodeWorkerHandler,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    MultimodalPrefillWorkerHandler,
)


def _request(objects: list[dict], *, mode: str = "enforce") -> PreprocessedRequest:
    return PreprocessedRequest(
        token_ids=[1, 2, 3],
        stop_conditions=StopConditions(),
        sampling_options=SamplingOptions(),
        multi_modal_data={
            "image_url": [{"Url": f"https://x/{i}.jpg"} for i in range(len(objects))]
        },
        mm_routing_info={
            "epd_prefill_selection": {"mode": mode, "worker_id": 91},
            "epd_routing_plan": {"objects": objects},
        },
    )


def test_parse_complete_enforced_plan():
    request = _request(
        [
            {
                "object_index": 0,
                "source_kind": "P_LOCAL",
                "source_worker_id": 91,
                "source_worker_generation": 1,
            },
            {
                "object_index": 1,
                "source_kind": "E_COMPUTE",
                "source_worker_id": 7,
                "source_worker_generation": 2,
            },
        ]
    )
    planned = parse_planned_media_objects(request, available_encode_worker_ids={7, 8})
    assert planned is not None
    assert [item.plan.source_kind for item in planned] == [
        MMSourceKind.P_LOCAL,
        MMSourceKind.E_COMPUTE,
    ]
    assert planned[0].cache_key == sglang_image_cache_key("https://x/0.jpg")
    assert target_prefill_worker(request) == 91


def test_stale_worker_plan_is_preserved_for_retry():
    request = _request(
        [
            {
                "object_index": 0,
                "source_kind": "E_CACHE",
                "source_worker_id": 7,
                "source_worker_generation": 1,
            }
        ]
    )
    planned = parse_planned_media_objects(request, available_encode_worker_ids={8})
    assert planned is not None
    assert planned[0].plan.source_worker_id == 7
    request.mm_routing_info["epd_prefill_selection"]["mode"] = "observe"  # type: ignore[index]
    assert parse_planned_media_objects(request, available_encode_worker_ids={7}) is None


def test_response_indices_must_be_complete_and_unique():
    request = _request(
        [
            {
                "object_index": 0,
                "source_kind": "E_COMPUTE",
                "source_worker_id": 7,
                "source_worker_generation": 1,
            },
            {
                "object_index": 1,
                "source_kind": "E_COMPUTE",
                "source_worker_id": 7,
                "source_worker_generation": 1,
            },
        ]
    )
    planned = parse_planned_media_objects(request, available_encode_worker_ids={7})
    assert planned is not None
    validate_object_response_indices(planned, [1, 0])

    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        validate_object_response_indices(planned, [0, 0])


def _object_plan(
    source_workers: list[int], *, source_kind: str = "E_COMPUTE"
) -> list[dict]:
    return [
        {
            "object_index": index,
            "source_kind": source_kind,
            "source_worker_id": worker_id,
            "source_worker_generation": 1,
        }
        for index, worker_id in enumerate(source_workers)
    ]


def _multimodal_request(objects: list[dict]) -> SglangMultimodalRequest:
    token_ids = [10]
    for index in range(len(objects)):
        token_ids.extend([99, 20 + index])
    return SglangMultimodalRequest(
        request=_request(objects).model_copy(update={"token_ids": token_ids})
    )


def _transfer(object_index: int, shape: tuple[int, int]) -> TransferRequest:
    return TransferRequest(
        embeddings_shape=list(shape),
        embedding_dtype_str="float32",
        serialized_request={"object_index": object_index},
    )


def _part(
    object_index: int,
    *,
    shape: tuple[int, int],
    modality: str = "IMAGE",
    cache_key: str | None = None,
) -> SglangEpdObjectPart:
    return SglangEpdObjectPart(
        object_index=object_index,
        modality=modality,
        cache_key=cache_key or sglang_image_cache_key(f"https://x/{object_index}.jpg"),
        embeddings_shape=shape,
        transfer_payload=_transfer(object_index, shape),
        grid_thw=[1, 1, shape[0]],
        num_mm_tokens=shape[0],
    )


class _EmbeddingReceiver:
    def __init__(
        self,
        tensors: dict[int, torch.Tensor],
        failures: set[int] | None = None,
    ) -> None:
        self.tensors = tensors
        self.failures = failures or set()

    async def receive_embeddings(self, request: TransferRequest):
        object_index = request.serialized_request["object_index"]
        if object_index in self.failures:
            raise TimeoutError(f"transfer {object_index} timed out")
        return object_index, self.tensors[object_index]


class _EmbeddingsProcessor:
    def __init__(
        self,
        tensors: dict[int, torch.Tensor],
        failures: set[int] | None = None,
    ) -> None:
        self.embedding_receiver = _EmbeddingReceiver(tensors, failures)
        self.released: list[int] = []

    def release_embeddings(self, tensor_id: int) -> None:
        self.released.append(tensor_id)

    def create_multimodal_image_item(self, embeddings, image_grid_thw):
        return {"embeddings": embeddings, "image_grid_thw": image_grid_thw}

    def create_multimodal_video_item(
        self,
        embeddings,
        video_grid_thw,
        second_per_grid_ts=None,
        video_timestamps=None,
    ):
        return {
            "embeddings": embeddings,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            "video_timestamps": video_timestamps,
        }


class _EncodeWorkerClient:
    def __init__(
        self,
        parts_by_worker: dict[int, list[SglangEpdObjectPart]],
        *,
        instance_ids: set[int] | None = None,
        fail_workers: set[int] | None = None,
    ) -> None:
        self.parts_by_worker = parts_by_worker
        self._instance_ids = instance_ids or set(parts_by_worker)
        self.fail_workers = fail_workers or set()
        self.calls: list[tuple[int, dict]] = []
        self.streams: list[_TrackedResponseStream] = []

    def instance_ids(self):
        return list(self._instance_ids)

    async def direct(self, payload: str, worker_id: int, context=None):
        self.calls.append((worker_id, json.loads(payload)))
        if worker_id in self.fail_workers:
            raise RuntimeError(f"worker {worker_id} disappeared")

        stream = _TrackedResponseStream(
            [
                SglangEpdObjectResponse(
                    parts=self.parts_by_worker[worker_id]
                ).model_dump(mode="json"),
                {"response_type": "trailer"},
            ]
        )
        self.streams.append(stream)
        return stream


class _TrackedResponseStream:
    def __init__(self, items: list[dict]) -> None:
        self.items = iter(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.closed = True


def _prefill_handler(
    client: _EncodeWorkerClient,
    tensors: dict[int, torch.Tensor],
    *,
    cache: MultimodalEmbeddingCacheManager | None = None,
    receive_failures: set[int] | None = None,
) -> MultimodalPrefillWorkerHandler:
    handler = object.__new__(MultimodalPrefillWorkerHandler)
    handler.encode_worker_client = client
    handler.embeddings_processor = _EmbeddingsProcessor(tensors, receive_failures)
    handler._embedding_cache = cache
    handler._cache_publisher = None
    handler.image_token_id = 99
    handler.video_token_id = 100
    return handler


@pytest.mark.asyncio
async def test_coordinate_epd_objects_groups_workers_and_restores_object_order():
    tensors = {
        0: torch.full((1, 2), 10.0),
        1: torch.full((2, 2), 20.0),
        2: torch.full((3, 2), 30.0),
    }
    client = _EncodeWorkerClient(
        {
            7: [_part(1, shape=(2, 2))],
            8: [_part(2, shape=(3, 2)), _part(0, shape=(1, 2))],
        }
    )
    handler = _prefill_handler(client, tensors)
    request = _multimodal_request(_object_plan([8, 7, 8]))

    image_items, video_items = await handler._coordinate_epd_objects(request, None)

    assert video_items == []
    assert {worker_id for worker_id, _ in client.calls} == {7, 8}
    requested_by_worker = {
        worker_id: [item["object_index"] for item in payload["objects"]]
        for worker_id, payload in client.calls
    }
    assert requested_by_worker == {7: [1], 8: [0, 2]}
    assert torch.equal(
        image_items[0]["embeddings"],
        torch.cat([tensors[0], tensors[1], tensors[2]], dim=0),
    )
    assert request.request.token_ids == [
        10,
        99,
        20,
        99,
        99,
        21,
        99,
        99,
        99,
        22,
    ]
    assert sorted(handler.embeddings_processor.released) == [0, 1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ([_part(0, shape=(1, 2)), _part(0, shape=(1, 2))], "duplicate"),
        ([_part(0, shape=(1, 2))], "omitted or added"),
        (
            [
                _part(0, shape=(1, 2), modality="VIDEO"),
                _part(1, shape=(1, 2)),
            ],
            "modality",
        ),
        (
            [
                _part(0, shape=(1, 2), cache_key="wrong"),
                _part(1, shape=(1, 2)),
            ],
            "cache key",
        ),
        ([_part(0, shape=(2, 2)), _part(1, shape=(1, 2))], "shape"),
    ],
)
async def test_coordinate_epd_objects_rejects_invalid_responses(parts, message):
    tensors = {0: torch.ones((1, 2)), 1: torch.ones((1, 2))}
    handler = _prefill_handler(_EncodeWorkerClient({7: parts}), tensors)
    request = _multimodal_request(_object_plan([7, 7]))

    with pytest.raises(ValueError, match=message):
        await handler._coordinate_epd_objects(request, None)


@pytest.mark.asyncio
async def test_coordinate_epd_objects_uses_local_hit_and_refetches_stale_local_miss():
    cache = MultimodalEmbeddingCacheManager(1024 * 1024)
    cached = CachedEmbedding(tensor=torch.full((2, 2), 11.0), image_grid_thw=[1, 1, 2])
    cache.set(sglang_image_cache_key("https://x/0.jpg"), cached)
    tensors = {1: torch.full((1, 2), 22.0)}
    client = _EncodeWorkerClient({5: [_part(1, shape=(1, 2))]}, instance_ids={5, 9})
    handler = _prefill_handler(client, tensors, cache=cache)
    request = _multimodal_request(_object_plan([91, 91], source_kind="P_LOCAL"))

    image_items, _ = await handler._coordinate_epd_objects(request, None)

    assert [worker_id for worker_id, _ in client.calls] == [5]
    assert client.calls[0][1]["objects"][0]["object_index"] == 1
    assert torch.equal(
        image_items[0]["embeddings"], torch.cat([cached.tensor, tensors[1]], dim=0)
    )
    assert cache.get(sglang_image_cache_key("https://x/1.jpg")) is not None


@pytest.mark.asyncio
async def test_coordinate_epd_objects_retries_url_objects_when_planned_worker_dies():
    tensors = {0: torch.full((1, 2), 33.0)}
    client = _EncodeWorkerClient(
        {9: [_part(0, shape=(1, 2))]},
        instance_ids={7, 9},
        fail_workers={7},
    )
    handler = _prefill_handler(client, tensors)
    request = _multimodal_request(_object_plan([7]))

    image_items, _ = await handler._coordinate_epd_objects(request, None)

    assert [worker_id for worker_id, _ in client.calls] == [7, 9]
    assert torch.equal(image_items[0]["embeddings"], tensors[0])


@pytest.mark.asyncio
async def test_coordinate_epd_objects_retries_stale_unregistered_worker():
    tensors = {0: torch.full((1, 2), 44.0)}
    client = _EncodeWorkerClient(
        {9: [_part(0, shape=(1, 2))]},
        instance_ids={9},
        fail_workers={7},
    )
    handler = _prefill_handler(client, tensors)
    request = _multimodal_request(_object_plan([7]))

    image_items, _ = await handler._coordinate_epd_objects(request, None)

    assert [worker_id for worker_id, _ in client.calls] == [7, 9]
    assert torch.equal(image_items[0]["embeddings"], tensors[0])


@pytest.mark.asyncio
async def test_coordinate_epd_objects_fails_explicitly_for_stale_uuid_p_local():
    client = _EncodeWorkerClient({}, instance_ids={9})
    handler = _prefill_handler(client, {})
    request = _multimodal_request(_object_plan([91], source_kind="P_LOCAL"))
    request.request.multi_modal_data = {"image_url": [{"UuidOnly": "cached-image"}]}

    with pytest.raises(RuntimeError, match="UUID-only object 0 disappeared"):
        await handler._coordinate_epd_objects(request, None)

    assert client.calls == []


@pytest.mark.asyncio
async def test_coordinate_epd_objects_does_not_recompute_stale_uuid_e_cache():
    client = _EncodeWorkerClient({}, instance_ids={9}, fail_workers={7})
    handler = _prefill_handler(client, {})
    request = _multimodal_request(_object_plan([7], source_kind="E_CACHE"))
    request.request.multi_modal_data = {"image_url": [{"UuidOnly": "cached-image"}]}

    with pytest.raises(RuntimeError, match="UUID-only embedding holder disappeared"):
        await handler._coordinate_epd_objects(request, None)

    assert [worker_id for worker_id, _ in client.calls] == [7]


@pytest.mark.asyncio
async def test_coordinate_epd_objects_closes_streams_after_transfer_timeout():
    tensors = {0: torch.ones((1, 2)), 1: torch.ones((1, 2))}
    client = _EncodeWorkerClient(
        {
            7: [_part(0, shape=(1, 2))],
            8: [_part(1, shape=(1, 2))],
        }
    )
    handler = _prefill_handler(client, tensors, receive_failures={1})
    request = _multimodal_request(_object_plan([7, 8]))

    with pytest.raises(TimeoutError, match="transfer 1 timed out"):
        await handler._coordinate_epd_objects(request, None)

    assert all(stream.closed for stream in client.streams)
    assert handler.embeddings_processor.released == [0]


@pytest.mark.asyncio
async def test_encode_object_response_waits_for_all_transfer_futures():
    handler = object.__new__(MultimodalEncodeWorkerHandler)
    futures = [asyncio.get_running_loop().create_future() for _ in range(2)]

    async def encode_object(media_object, modality):
        return _part(media_object.object_index, shape=(1, 2)), futures[
            media_object.object_index
        ]

    handler._encode_epd_object = encode_object
    request = SglangEpdObjectRequest(
        objects=[
            {
                "object_index": index,
                "modality": "IMAGE",
                "url": f"https://x/{index}.jpg",
                "expected_cache_key": sglang_image_cache_key(f"https://x/{index}.jpg"),
            }
            for index in range(2)
        ]
    )
    responses = handler._generate_epd_objects(request.model_dump(mode="json"))

    first = await anext(responses)
    assert [part["object_index"] for part in first["parts"]] == [0, 1]

    completion = asyncio.create_task(anext(responses))
    await asyncio.sleep(0)
    assert not completion.done()
    for future in futures:
        future.set_result(None)
    with pytest.raises(StopAsyncIteration):
        await completion


@pytest.mark.asyncio
async def test_prefill_bootstrap_returns_expanded_input_ids():
    handler = object.__new__(MultimodalPrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 9876
    handler._generate_bootstrap_room = lambda: 42

    async def coordinate(request, context):
        request.request.token_ids = [1, 99, 99, 2]
        return ([{"image_data": "ready"}], [])

    processed = []

    async def process(request, room, context=None, prepared_mm_items=None):
        processed.append((room, prepared_mm_items))

    handler._coordinate_epd_objects = coordinate
    handler._process_prefill_generation = process
    disagg_request = DisaggSglangMultimodalRequest(
        request=_multimodal_request(_object_plan([7])), sampling_params={}
    )

    responses = [item async for item in handler.generate(disagg_request, None)]

    assert json.loads(responses[0])["input_ids"] == [1, 99, 99, 2]
    assert processed == [(42, ([{"image_data": "ready"}], []))]
