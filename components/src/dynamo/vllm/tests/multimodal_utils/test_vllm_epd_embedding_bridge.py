# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from dynamo.vllm.multimodal_utils.epd_embedding_bridge import (
    VllmEpdEmbeddingBridge,
    atomic_write_json,
    read_json,
    request_directory,
    write_cache_event,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


async def _wait_for(path, timeout_s=1.0):
    async with asyncio.timeout(timeout_s):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_bridge_remaps_subset_plan_and_preserves_object_order(tmp_path):
    loader = SimpleNamespace(load_multimodal_embedding_parts=AsyncMock())
    loader.load_multimodal_embedding_parts.return_value = [
        SimpleNamespace(tensor=torch.full((2, 3), 2.0)),
        SimpleNamespace(tensor=torch.full((2, 3), 0.0)),
    ]
    bridge = VllmEpdEmbeddingBridge(tmp_path, loader)
    routing_plan = {
        "target_p_worker_id": 7,
        "objects": [
            {"object_index": index, "source_worker_id": index + 10}
            for index in range(3)
        ],
    }

    try:
        await bridge.register(
            request_id="request-1",
            image_urls=["url-0", "url-1", "url-2"],
            identifiers=["hash-0", "hash-1", "hash-2"],
            model="model",
            routing_plan=routing_plan,
            context=object(),
        )
        directory = request_directory(tmp_path, "request-1")
        manifest = read_json(directory / "manifest.json")
        atomic_write_json(
            directory / "need.json",
            {
                "request_id": "request-1",
                "generation": manifest["generation"],
                "items": [
                    {"index": 2, "identifier": "hash-2"},
                    {"index": 0, "identifier": "hash-0"},
                ],
            },
        )

        await _wait_for(directory / "ready.json")
        call = loader.load_multimodal_embedding_parts.await_args
        assert call.args[:2] == (["url-2", "url-0"], "request-1")
        assert call.kwargs["routing_plan"] == {
            "target_p_worker_id": 7,
            "objects": [
                {"object_index": 0, "source_worker_id": 12},
                {"object_index": 1, "source_worker_id": 10},
            ],
        }
        ready = read_json(directory / "ready.json")
        assert ready["generation"] == manifest["generation"]
        assert [item["index"] for item in ready["items"]] == [2, 0]
        assert [item["identifier"] for item in ready["items"]] == [
            "hash-2",
            "hash-0",
        ]
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_bridge_rejects_mismatched_identity(tmp_path):
    loader = SimpleNamespace(load_multimodal_embedding_parts=AsyncMock())
    bridge = VllmEpdEmbeddingBridge(tmp_path, loader)

    try:
        await bridge.register(
            request_id="request-1",
            image_urls=["url-0"],
            identifiers=["hash-0"],
            model="model",
            routing_plan=None,
            context=None,
        )
        directory = request_directory(tmp_path, "request-1")
        manifest = read_json(directory / "manifest.json")
        atomic_write_json(
            directory / "need.json",
            {
                "request_id": "request-1",
                "generation": manifest["generation"],
                "items": [{"index": 0, "identifier": "wrong-hash"}],
            },
        )

        await _wait_for(directory / "error.json")
        error = read_json(directory / "error.json")
        assert error["generation"] == manifest["generation"]
        assert "identity mismatch" in error["message"]
        loader.load_multimodal_embedding_parts.assert_not_awaited()
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_register_rejects_duplicate_identifiers(tmp_path):
    bridge = VllmEpdEmbeddingBridge(
        tmp_path,
        SimpleNamespace(load_multimodal_embedding_parts=AsyncMock()),
    )
    try:
        with pytest.raises(ValueError, match="identifiers must be unique"):
            await bridge.register(
                request_id="request-1",
                image_urls=["url-0", "url-1"],
                identifiers=["same", "same"],
                model="model",
                routing_plan=None,
                context=None,
            )
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_unregister_cancels_fetch_and_removes_request_files(tmp_path):
    started = asyncio.Event()

    async def load(*args, **kwargs):
        started.set()
        await asyncio.Future()

    loader = SimpleNamespace(
        load_multimodal_embedding_parts=AsyncMock(side_effect=load)
    )
    bridge = VllmEpdEmbeddingBridge(tmp_path, loader)

    try:
        await bridge.register(
            request_id="request-1",
            image_urls=["url-0"],
            identifiers=["hash-0"],
            model="model",
            routing_plan=None,
            context=None,
        )
        directory = request_directory(tmp_path, "request-1")
        manifest = read_json(directory / "manifest.json")
        atomic_write_json(
            directory / "need.json",
            {
                "request_id": "request-1",
                "generation": manifest["generation"],
                "items": [{"index": 0, "identifier": "hash-0"}],
            },
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await bridge.unregister("request-1")

        assert not directory.exists()
        assert not bridge._fetch_tasks
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_new_registration_replaces_stale_generation(tmp_path):
    bridge = VllmEpdEmbeddingBridge(
        tmp_path,
        SimpleNamespace(load_multimodal_embedding_parts=AsyncMock()),
    )
    try:
        await bridge.register(
            request_id="request-1",
            image_urls=["url-0"],
            identifiers=["hash-0"],
            model="model",
            routing_plan=None,
            context=None,
        )
        directory = request_directory(tmp_path, "request-1")
        first = read_json(directory / "manifest.json")["generation"]
        atomic_write_json(
            directory / "ready.json",
            {"generation": first, "items": []},
        )

        await bridge.register(
            request_id="request-1",
            image_urls=["url-0"],
            identifiers=["hash-0"],
            model="model",
            routing_plan=None,
            context=None,
        )

        second = read_json(directory / "manifest.json")["generation"]
        assert second != first
        assert not (directory / "ready.json").exists()
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_cache_events_publish_only_after_worker_mutation(tmp_path):
    publisher = MagicMock()
    bridge = VllmEpdEmbeddingBridge(
        tmp_path,
        SimpleNamespace(load_multimodal_embedding_parts=AsyncMock()),
        publisher,
    )
    try:
        write_cache_event(tmp_path, "ADD", "hash-0")
        write_cache_event(tmp_path, "REMOVE", "hash-1")

        async with asyncio.timeout(1.0):
            while publisher.publish_delta.call_count != 2:
                await asyncio.sleep(0.01)

        assert {
            (tuple(call.args[0]), tuple(call.args[1]))
            for call in publisher.publish_delta.call_args_list
        } == {
            (("hash-0",), ()),
            ((), ("hash-1",)),
        }
        assert not list((tmp_path / "events").glob("*.json"))
    finally:
        await bridge.close()
