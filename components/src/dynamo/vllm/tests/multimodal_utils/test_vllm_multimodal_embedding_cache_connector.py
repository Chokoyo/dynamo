# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.vllm.multimodal_utils import cache_config as cache_config_mod
from dynamo.vllm.multimodal_utils import multimodal_embedding_cache_connector as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_vllm_config(
    capacity_gb: float = 1.0, bridge_dir: str | None = None
) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }
    if bridge_dir is not None:
        config.ec_transfer_config.ec_connector_extra_config["epd_bridge_dir"] = (
            bridge_dir
        )
    config.model_config.get_hidden_size.return_value = 4096
    config.model_config.dtype = torch.float16
    return config


class TestCacheConfiguration:
    def test_disabled_capacity_leaves_engine_args_unchanged(self):
        engine_args = SimpleNamespace()

        cache_config_mod.configure_multimodal_embedding_cache(
            engine_args,
            route_to_encoder=False,
            capacity_gb=0,
            namespace="dynamo",
            component="backend",
        )

        assert not hasattr(engine_args, "ec_transfer_config")

    def test_encoder_routing_configures_post_kv_connector(self):
        engine_args = SimpleNamespace()
        transfer_config = object()

        with (
            patch.dict("os.environ", {"DYN_VLLM_EPD_BRIDGE_DIR": "/tmp/bridge"}),
            patch("vllm.config.ECTransferConfig", return_value=transfer_config) as cls,
        ):
            cache_config_mod.configure_multimodal_embedding_cache(
                engine_args,
                route_to_encoder=True,
                capacity_gb=1,
                namespace="dynamo",
                component="backend",
            )

        assert engine_args.ec_transfer_config is transfer_config
        assert cls.call_args.kwargs["ec_connector_extra_config"] == {
            "multimodal_embedding_cache_capacity_gb": 1,
            "epd_bridge_dir": "/tmp/bridge",
        }

    def test_generated_bridge_dir_is_inherited_by_engine_core(self):
        first_engine_args = SimpleNamespace()
        second_engine_args = SimpleNamespace()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(cache_config_mod.tempfile, "gettempdir", return_value="/tmp"),
            patch.object(cache_config_mod.os, "getpid", return_value=1234) as getpid,
            patch("vllm.config.ECTransferConfig") as cls,
        ):
            cache_config_mod.configure_multimodal_embedding_cache(
                first_engine_args,
                route_to_encoder=True,
                capacity_gb=1,
                namespace="dynamo",
                component="backend",
            )
            cache_config_mod.configure_multimodal_embedding_cache(
                second_engine_args,
                route_to_encoder=True,
                capacity_gb=1,
                namespace="dynamo",
                component="backend",
            )

            expected = "/tmp/dynamo-vllm-epd-1234-dynamo-backend"
            assert os.environ["DYN_VLLM_EPD_BRIDGE_DIR"] == expected
            assert [
                call.kwargs["ec_connector_extra_config"]["epd_bridge_dir"]
                for call in cls.call_args_list
            ] == [expected, expected]
            getpid.assert_called_with()

    def test_enabled_capacity_configures_dynamo_connector(self):
        engine_args = SimpleNamespace()
        transfer_config = object()

        with (
            patch.dict("os.environ", {"DYN_VLLM_EPD_BRIDGE_DIR": "/tmp/bridge"}),
            patch("vllm.config.ECTransferConfig", return_value=transfer_config) as cls,
        ):
            cache_config_mod.configure_multimodal_embedding_cache(
                engine_args,
                route_to_encoder=False,
                capacity_gb=2.5,
                namespace="deployment",
                component="prefill",
            )

        assert engine_args.ec_transfer_config is transfer_config
        cls.assert_called_once_with(
            engine_id="deployment.prefill.backend.0",
            ec_role="ec_both",
            ec_connector="DynamoMultimodalEmbeddingCacheConnector",
            ec_connector_module_path=(
                "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
            ),
            ec_connector_extra_config={
                "multimodal_embedding_cache_capacity_gb": 2.5,
                "epd_bridge_dir": "/tmp/bridge",
            },
        )


class TestVersionCheck:
    def test_warns_old_vllm(self):
        with (
            patch.object(mod, "_vllm_version", "0.16.5"),
            patch.object(mod.ECConnectorBase, "__init__", return_value=None),
            patch.object(mod.logger, "warning") as mock_warn,
        ):
            connector = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(),
                role=MagicMock(),
            )
            assert connector is not None
            mock_warn.assert_called_once()
            assert mock_warn.call_args[0][1] == mod.MINIMUM_VLLM_VERSION
            assert mock_warn.call_args[0][2] == "0.16.5"


class TestSchedulerSideLRU:
    """Test the scheduler-side logical LRU cache and metadata generation."""

    def _make_connector(self, capacity_gb: float = 1.0):
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            return mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(capacity_gb),
                role=MagicMock(),
            )

    def _make_request(
        self,
        hashes_and_embeds: list[tuple[str, int]],
        *,
        request_id: str = "request-1",
        offsets: list[tuple[int, int]] | None = None,
    ) -> MagicMock:
        request = MagicMock()
        request.request_id = request_id
        features = []
        for index, (h, _) in enumerate(hashes_and_embeds):
            f = MagicMock()
            f.identifier = h
            offset, length = offsets[index] if offsets is not None else (0, 100)
            f.mm_position = SimpleNamespace(offset=offset, length=length)
            features.append(f)
        request.mm_features = features

        def get_num_encoder_embeds(idx):
            return hashes_and_embeds[idx][1]

        request.get_num_encoder_embeds = get_num_encoder_embeds
        return request

    def test_has_cache_item_miss_then_hit(self):
        conn = self._make_connector()
        opaque_uuid = "catalog/image:v2"
        assert not conn.has_cache_item(opaque_uuid)

        request = self._make_request([(opaque_uuid, 100)])
        conn.update_state_after_alloc(request, 0)

        with patch.object(mod.logger, "debug") as log_debug:
            assert conn.has_cache_item(opaque_uuid)
        log_debug.assert_called_once_with(
            mod.EMBEDDING_CACHE_HIT_LOG,
            opaque_uuid,
        )

    def test_update_state_plans_save(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(request, 0)

        scheduler_output = MagicMock()
        meta = conn.build_connector_meta(scheduler_output)
        assert isinstance(meta, mod.MultimodalEmbeddingCacheConnectorMetadata)
        assert "hash_a" in meta.saves
        assert meta.loads == []
        assert meta.evicts == []

    def test_update_state_plans_load_for_cached(self):
        conn = self._make_connector()
        request = self._make_request([("hash_a", 100)])

        conn.update_state_after_alloc(request, 0)
        conn.build_connector_meta(MagicMock())

        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())
        assert "hash_a" in meta.loads
        assert meta.saves == []

    def test_eviction_under_pressure(self):
        # 4096 hidden_size * 2 bytes (fp16) = 8192 bytes per embed
        conn = self._make_connector()
        bpe = conn._bytes_per_embed  # 8192
        # Set capacity to hold exactly 200 embeds worth of bytes
        conn._capacity_bytes = 200 * bpe

        req_a = self._make_request([("hash_a", 100)])
        conn.update_state_after_alloc(req_a, 0)
        conn.build_connector_meta(MagicMock())

        req_b = self._make_request([("hash_b", 100)])
        conn.update_state_after_alloc(req_b, 0)
        conn.build_connector_meta(MagicMock())

        assert conn._num_used_bytes == 200 * bpe

        # Adding hash_c (100 embeds) should evict hash_a (LRU)
        req_c = self._make_request([("hash_c", 100)])
        conn.update_state_after_alloc(req_c, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert "hash_c" in meta.saves
        assert "hash_a" in meta.evicts
        assert "hash_a" not in conn._cache_order
        assert "hash_c" in conn._cache_order

    def test_skip_oversized_item(self):
        conn = self._make_connector()
        bpe = conn._bytes_per_embed
        conn._capacity_bytes = 50 * bpe

        request = self._make_request([("huge_hash", 100)])
        conn.update_state_after_alloc(request, 0)
        meta = conn.build_connector_meta(MagicMock())

        assert meta.saves == []
        assert meta.loads == []
        assert "huge_hash" not in conn._cache_order

    def test_full_kv_coverage_does_not_request_embedding(self, tmp_path):
        bridge_dir = str(tmp_path)
        conn = self._make_connector()
        conn._bridge_dir = tmp_path
        request = self._make_request(
            [("hash_a", 10)],
            offsets=[(5, 10)],
        )
        directory = mod.request_directory(bridge_dir, request.request_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )

        assert conn.ensure_cache_available(request, num_computed_tokens=15)
        assert not (directory / "need.json").exists()

    def test_uncovered_embedding_is_loaded_from_bridge_after_kv_lookup(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        request = self._make_request(
            [("hash_a", 4)],
            offsets=[(8, 4)],
        )
        directory = mod.request_directory(bridge_dir, request.request_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )

        assert not scheduler.ensure_cache_available(request, num_computed_tokens=8)
        assert mod.read_json(directory / "need.json")["items"] == [
            {"identifier": "hash_a", "index": 0}
        ]
        assert mod.read_json(directory / "need.json")["generation"] == "generation-1"

        tensor_path = directory / "item-0.pt"
        torch.save(torch.ones(1, 4, 8), tensor_path)
        mod.atomic_write_json(
            directory / "ready.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "items": [
                    {
                        "index": 0,
                        "identifier": "hash_a",
                        "path": str(tensor_path),
                        "shape": [1, 4, 8],
                        "dtype": "torch.float32",
                    }
                ],
            },
        )

        assert scheduler.ensure_cache_available(request, num_computed_tokens=8)
        scheduler.update_state_after_alloc(request, 0)
        metadata = scheduler.build_connector_meta(MagicMock())
        assert metadata.loads == ["hash_a"]
        assert metadata.external_loads == {"hash_a": (str(tensor_path), 4)}

        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            worker = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        worker._device = "cpu"
        worker.bind_connector_metadata(metadata)
        encoder_cache = {}
        worker.start_load_caches(encoder_cache)

        assert encoder_cache["hash_a"].shape == (4, 8)
        events = list((tmp_path / "events").glob("*.json"))
        assert len(events) == 1
        assert mod.read_json(events[0]) == {
            "action": "ADD",
            "cache_key": "hash_a",
        }

    def test_randomized_engine_request_id_uses_external_bridge_id(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        request = self._make_request(
            [("hash_a", 4)],
            request_id="request-1-randomized",
            offsets=[(8, 4)],
        )
        request.external_req_id = "request-1"
        directory = mod.request_directory(bridge_dir, request.external_req_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.external_req_id,
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )

        assert not scheduler.ensure_cache_available(request, num_computed_tokens=8)
        assert mod.read_json(directory / "need.json") == {
            "request_id": request.external_req_id,
            "generation": "generation-1",
            "items": [{"identifier": "hash_a", "index": 0}],
        }
        assert request.external_req_id in scheduler._pending_since
        assert request.request_id not in scheduler._pending_since

        scheduler.request_finished(request)
        assert request.external_req_id not in scheduler._pending_since

    def test_randomized_engine_request_id_recovers_from_bridge_manifest(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        request = self._make_request(
            [("hash_a", 4)],
            request_id="request-1-deadbeef",
            offsets=[(8, 4)],
        )
        directory = mod.request_directory(bridge_dir, "request-1")
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": "request-1",
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )

        assert not scheduler.ensure_cache_available(request, num_computed_tokens=8)
        assert mod.read_json(directory / "need.json") == {
            "request_id": "request-1",
            "generation": "generation-1",
            "items": [{"identifier": "hash_a", "index": 0}],
        }
        assert "request-1" in scheduler._pending_since

    def test_stale_ready_generation_is_ignored(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        request = self._make_request([("hash_a", 4)])
        directory = mod.request_directory(bridge_dir, request.request_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.request_id,
                "generation": "generation-new",
                "identifiers": ["hash_a"],
            },
        )
        stale_tensor = directory / "item-stale.pt"
        torch.save(torch.ones(4, 8), stale_tensor)
        mod.atomic_write_json(
            directory / "ready.json",
            {
                "request_id": request.request_id,
                "generation": "generation-old",
                "items": [
                    {
                        "index": 0,
                        "identifier": "hash_a",
                        "path": str(stale_tensor),
                    }
                ],
            },
        )

        assert not scheduler.ensure_cache_available(request, num_computed_tokens=0)
        assert not (directory / "ready.json").exists()
        assert mod.read_json(directory / "need.json")["generation"] == "generation-new"
        assert "hash_a" not in scheduler._cache_order

    def test_matching_error_generation_fails_request(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        request = self._make_request([("hash_a", 4)])
        directory = mod.request_directory(bridge_dir, request.request_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )
        mod.atomic_write_json(
            directory / "error.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "message": "all holders disappeared",
            },
        )

        with pytest.raises(RuntimeError, match="all holders disappeared"):
            scheduler.ensure_cache_available(request, num_computed_tokens=0)

    def test_post_kv_fetch_timeout_is_explicit(self, tmp_path):
        bridge_dir = str(tmp_path)
        with patch.object(mod.ECConnectorBase, "__init__", return_value=None):
            scheduler = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(1.0, bridge_dir),
                role=MagicMock(),
            )
        scheduler._fetch_timeout_s = -1
        request = self._make_request([("hash_a", 4)])
        directory = mod.request_directory(bridge_dir, request.request_id)
        mod.atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request.request_id,
                "generation": "generation-1",
                "identifiers": ["hash_a"],
            },
        )

        with pytest.raises(TimeoutError, match=request.request_id):
            scheduler.ensure_cache_available(request, num_computed_tokens=0)
