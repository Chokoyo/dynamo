# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for load_multimodal_embeddings in prefill_worker_utils."""

from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    TransferRequest,
)
from dynamo.common.multimodal_epd import MMObjectPlan, MMSourceKind
from dynamo.vllm.multimodal_utils import prefill_worker_utils as mod
from dynamo.vllm.multimodal_utils.protocol import MultiModalGroup, MultiModalInput

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

MODEL = "test-model"
DTYPE = torch.float16


def _routing_plan(*objects: dict) -> dict:
    return {"objects": list(objects)}


def _object_plan(index: int, worker_id: int, source: str = "E_COMPUTE") -> dict:
    return {
        "object_index": index,
        "source_kind": source,
        "source_worker_id": worker_id,
        "source_worker_generation": 1,
        "estimated_cost_ms": 0.0,
    }


class _Response:
    def __init__(self, payload: str):
        self._payload = payload

    def data(self) -> str:
        return self._payload


class _Receiver(AbstractEmbeddingReceiver):
    async def receive_embeddings(
        self, request: TransferRequest
    ) -> tuple[int, torch.Tensor]:
        return int(request.serialized_request), torch.tensor(
            [[float(request.serialized_request)]], dtype=DTYPE
        )

    def release_tensor(self, tensor_id: int) -> None:
        pass


class _Client:
    def __init__(self):
        self.direct_calls: list[tuple[int, list[str]]] = []
        self.round_robin_calls = 0

    def instance_ids(self) -> list[int]:
        return [11, 22]

    async def direct(self, payload: str, worker_id: int, context=None):
        request = mod.vLLMMultimodalRequest.model_validate_json(payload)
        urls = [
            group.multimodal_input.image_url
            for group in request.multimodal_inputs or []
            if group.multimodal_input is not None
        ]
        self.direct_calls.append((worker_id, urls))

        async def stream():
            groups = []
            for url in urls:
                value = int(url.rsplit("/", 1)[-1])
                groups.append(
                    MultiModalGroup(
                        multimodal_input=MultiModalInput(image_url=url),
                        serialized_request=TransferRequest(
                            embeddings_shape=[1, 1],
                            embedding_dtype_str="float16",
                            serialized_request=value,
                        ),
                    )
                )
            response = request.model_copy(update={"multimodal_inputs": groups})
            yield _Response(response.model_dump_json())

        return stream()

    async def round_robin(self, payload: str, context=None):
        self.round_robin_calls += 1
        return await self.direct(payload, 11, context=context)


class _FailingDirectClient(_Client):
    async def direct(self, payload: str, worker_id: int, context=None):
        if worker_id == 22:
            request = mod.vLLMMultimodalRequest.model_validate_json(payload)
            urls = [
                group.multimodal_input.image_url
                for group in request.multimodal_inputs or []
                if group.multimodal_input is not None
            ]
            self.direct_calls.append((worker_id, urls))
            raise RuntimeError("planned worker disappeared")
        return await super().direct(payload, worker_id, context=context)


class _MismatchedDirectClient(_Client):
    async def direct(self, payload: str, worker_id: int, context=None):
        if worker_id != 22:
            return await super().direct(payload, worker_id, context=context)

        request = mod.vLLMMultimodalRequest.model_validate_json(payload)
        urls = [
            group.multimodal_input.image_url
            for group in request.multimodal_inputs or []
            if group.multimodal_input is not None
        ]
        self.direct_calls.append((worker_id, urls))

        async def stream():
            mismatched_group = MultiModalGroup(
                multimodal_input=MultiModalInput(image_url="http://image/999"),
                serialized_request=TransferRequest(
                    embeddings_shape=[1, 1],
                    embedding_dtype_str="float16",
                    serialized_request=999,
                ),
            )
            response = request.model_copy(
                update={"multimodal_inputs": [mismatched_group]}
            )
            yield _Response(response.model_dump_json())

        return stream()


class TestMultimodalEmbeddingLoader:
    @pytest.mark.asyncio
    async def test_direct_dispatch_restores_original_object_order(self):
        client = _Client()
        plans = [
            MMObjectPlan(0, MMSourceKind.E_COMPUTE, 22, 1, 0.0),
            MMObjectPlan(1, MMSourceKind.E_CACHE, 11, 1, 0.0),
            MMObjectPlan(2, MMSourceKind.E_COMPUTE, 22, 1, 0.0),
        ]

        groups, pending = await mod._fetch_from_encode_workers(
            client,
            ["http://image/1", "http://image/2", "http://image/3"],
            "req-direct",
            _Receiver(),
            object_plans=plans,
        )

        assert client.direct_calls == [
            (22, ["http://image/1", "http://image/3"]),
            (11, ["http://image/2"]),
        ]
        assert client.round_robin_calls == 0
        assert [group.loaded_embedding.item() for group in groups] == [1.0, 2.0, 3.0]
        assert pending is not None

    @pytest.mark.asyncio
    async def test_direct_failure_retries_on_another_encode_worker(self):
        client = _FailingDirectClient()

        groups, pending = await mod._fetch_from_encode_workers(
            client,
            ["http://image/1"],
            "req-retry-failure",
            _Receiver(),
            object_plans=[MMObjectPlan(0, MMSourceKind.E_CACHE, 22, 1, 0.0)],
        )

        assert client.direct_calls == [
            (22, ["http://image/1"]),
            (11, ["http://image/1"]),
        ]
        assert groups[0].multimodal_input.image_url == "http://image/1"
        assert groups[0].loaded_embedding.item() == 1.0
        assert pending is not None

    @pytest.mark.asyncio
    async def test_p_remote_dispatch_uses_prefill_embedding_client(self):
        encode_client = _Client()
        prefill_client = _Client()
        plan = _routing_plan(_object_plan(0, 22, source="P_REMOTE"))

        groups, pending = await mod._fetch_embeddings(
            encode_client,
            ["http://image/1"],
            "req-p-remote",
            _Receiver(),
            routing_plan=plan,
            prefill_worker_client=prefill_client,
        )

        assert prefill_client.direct_calls == [(22, ["http://image/1"])]
        assert encode_client.direct_calls == []
        assert groups[0].loaded_embedding.item() == 1.0
        assert pending is not None

    @pytest.mark.asyncio
    async def test_p_remote_failure_falls_back_to_encode_routing(self):
        encode_client = _Client()
        prefill_client = _FailingDirectClient()
        plan = _routing_plan(_object_plan(0, 22, source="P_REMOTE"))

        groups, pending = await mod._fetch_embeddings(
            encode_client,
            ["http://image/1"],
            "req-p-remote-fallback",
            _Receiver(),
            routing_plan=plan,
            prefill_worker_client=prefill_client,
        )

        assert prefill_client.direct_calls == [(22, ["http://image/1"])]
        assert encode_client.round_robin_calls == 1
        assert groups[0].loaded_embedding.item() == 1.0
        assert pending is not None

    @pytest.mark.asyncio
    async def test_p_remote_failure_only_falls_back_affected_source(self):
        encode_client = _Client()
        prefill_client = _FailingDirectClient()
        plan = _routing_plan(
            _object_plan(0, 11, source="P_REMOTE"),
            _object_plan(1, 22, source="P_REMOTE"),
        )

        groups, pending = await mod._fetch_embeddings(
            encode_client,
            ["http://image/1", "http://image/2"],
            "req-p-remote-partial-fallback",
            _Receiver(),
            routing_plan=plan,
            prefill_worker_client=prefill_client,
        )

        assert prefill_client.direct_calls == [
            (11, ["http://image/1"]),
            (22, ["http://image/2"]),
        ]
        assert encode_client.direct_calls == [(11, ["http://image/2"])]
        assert encode_client.round_robin_calls == 1
        assert [group.loaded_embedding.item() for group in groups] == [1.0, 2.0]
        assert pending is not None

    @pytest.mark.asyncio
    async def test_direct_identity_mismatch_retries_without_using_wrong_object(self):
        client = _MismatchedDirectClient()

        groups, pending = await mod._fetch_from_encode_workers(
            client,
            ["http://image/1"],
            "req-retry-mismatch",
            _Receiver(),
            object_plans=[MMObjectPlan(0, MMSourceKind.E_CACHE, 22, 1, 0.0)],
        )

        assert client.direct_calls == [
            (22, ["http://image/1"]),
            (11, ["http://image/1"]),
        ]
        assert groups[0].multimodal_input.image_url == "http://image/1"
        assert groups[0].loaded_embedding.item() == 1.0
        assert pending is not None

    @pytest.mark.asyncio
    async def test_unavailable_direct_worker_falls_back_to_legacy(self):
        client = _Client()
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(), loaded_embedding=tensor
        )
        plan = _routing_plan(_object_plan(0, 99))

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            groups, _ = await mod._fetch_embeddings(
                client,
                ["http://image/1"],
                "req-stale",
                _Receiver(),
                routing_plan=plan,
            )

        assert groups == [fake_group]
        assert mock_fetch.call_args.kwargs["object_plans"] is None

    @pytest.mark.asyncio
    async def test_mixed_p_local_and_direct_plan_filters_cached_object(self):
        client = _Client()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        cached_url = "http://image/cached"
        cached_tensor = torch.randn(1, 10, dtype=DTYPE)
        cache.set(
            mod.get_embedding_hash(cached_url),
            CachedEmbedding(tensor=cached_tensor, image_grid_thw=None),
        )
        remote_tensor = torch.randn(1, 10, dtype=DTYPE)
        remote_group = MultiModalGroup(
            multimodal_input=MultiModalInput(), loaded_embedding=remote_tensor
        )
        plan = _routing_plan(
            _object_plan(0, 101, "P_LOCAL"),
            _object_plan(1, 22, "E_CACHE"),
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([remote_group], None),
        ) as mock_fetch:
            groups, _ = await mod._fetch_embeddings(
                client,
                [cached_url, "http://image/remote"],
                "req-mixed",
                _Receiver(),
                cache=cache,
                routing_plan=plan,
            )

        passed_plans = mock_fetch.call_args.kwargs["object_plans"]
        assert passed_plans is not None
        assert [
            (item.object_index, item.source_worker_id) for item in passed_plans
        ] == [(1, 22)]
        assert torch.equal(groups[0].loaded_embedding, cached_tensor)
        assert torch.equal(groups[1].loaded_embedding, remote_tensor)

    @pytest.mark.asyncio
    async def test_all_cached(self):
        """All URLs cached -> no encode worker call, returns accumulated mm_data."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(1, 10, dtype=DTYPE)
        grid = [[1, 2, 3]]
        url = "http://img1.png"
        key = mod.get_embedding_hash(url)
        cache.set(key, CachedEmbedding(tensor=tensor, image_grid_thw=grid))

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_not_awaited()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_all_uncached_with_cache(self):
        """All URLs uncached with cache -> encode worker call, results cached."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

        key = mod.get_embedding_hash(url)
        cached = cache.get(key)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)

    @pytest.mark.asyncio
    async def test_cache_fill_publishes_residency_delta(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        published: list[tuple[list[str], list[str]]] = []
        publisher = SimpleNamespace(
            publish_delta=lambda added, removed: published.append((added, removed))
        )
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ):
            embedding_loader = mod.MultiModalEmbeddingLoader(
                AsyncMock(), None, cache, publisher
            )
            await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        assert published == [([mod.get_embedding_hash(url)], [])]

    @pytest.mark.asyncio
    async def test_no_cache(self):
        """Without cache -> all URLs go to encode workers."""
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, None)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_mixed_cache(self):
        """Mixed cache hits/misses -> only misses sent to encode workers."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        url_cached = "http://cached.png"
        url_miss = "http://miss.png"
        cached_tensor = torch.randn(1, 10, dtype=DTYPE)
        miss_tensor = torch.randn(1, 10, dtype=DTYPE)

        key = mod.get_embedding_hash(url_cached)
        cache.set(key, CachedEmbedding(tensor=cached_tensor, image_grid_thw=None))

        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=None,
            loaded_embedding=miss_tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url_cached, url_miss],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == [url_miss]
        expected = torch.cat((cached_tensor, miss_tensor))
        assert torch.equal(mm_data["image"], expected)
