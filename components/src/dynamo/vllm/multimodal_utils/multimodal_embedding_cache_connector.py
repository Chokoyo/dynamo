# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from packaging.version import Version
from vllm import __version__ as _vllm_version
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from .epd_embedding_bridge import (
    atomic_write_json,
    read_json,
    request_directory,
    write_cache_event,
    write_epd_audit_event,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

MINIMUM_VLLM_VERSION = "0.17.0"

# This connector runs inside vLLM's spawned EngineCore process, where Dynamo's
# logging bridge is unavailable. Use a vLLM child logger so
# VLLM_LOGGING_LEVEL controls these diagnostics.
logger = logging.getLogger("vllm.dynamo.multimodal_embedding_cache_connector")
EMBEDDING_CACHE_HIT_LOG = "Dynamo multimodal embedding cache hit: identifier=%r"


def _bridge_request_id(request: "Request", bridge_dir: Path | None = None) -> str:
    external_request_id = getattr(request, "external_req_id", None)
    if isinstance(external_request_id, str) and external_request_id:
        return external_request_id

    engine_request_id = request.request_id
    if bridge_dir is None:
        return engine_request_id

    if (request_directory(bridge_dir, engine_request_id) / "manifest.json").is_file():
        return engine_request_id

    candidate, separator, suffix = engine_request_id.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(character in "0123456789abcdef" for character in suffix.lower())
        and (request_directory(bridge_dir, candidate) / "manifest.json").is_file()
    ):
        return candidate

    return engine_request_id


def _get_device(vllm_config: "VllmConfig") -> str:
    device_config = getattr(vllm_config, "device_config", None)
    for field_name in ("device", "device_type"):
        device = getattr(device_config, field_name, None)
        if isinstance(device, torch.device):
            device = device.type
        if isinstance(device, str) and device in ("cuda", "xpu"):
            return device

    target_device = os.environ.get("VLLM_TARGET_DEVICE")
    if target_device in ("cuda", "xpu"):
        return target_device

    return "cuda"


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)
    external_loads: dict[str, tuple[str, int]] = field(default_factory=dict)


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, Tensor] on CPU and obeys commands
    without independent caching decisions.

    This mirrors vLLM's EncoderCacheManager pattern: the scheduler is the
    single source of truth for cache state; the worker is a plain dict storage.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        if Version(_vllm_version) < Version(MINIMUM_VLLM_VERSION):
            logger.warning(
                "DynamoMultimodalEmbeddingCacheConnector requires vLLM >= %s, "
                "but found %s. Some features may not work correctly.",
                MINIMUM_VLLM_VERSION,
                _vllm_version,
            )
        super().__init__(vllm_config=vllm_config, role=role)
        self._device = _get_device(vllm_config)

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError(
                "ec_transfer_config must be set for DynamoMultimodalEmbeddingCacheConnector"
            )

        extra_config = transfer_config.ec_connector_extra_config or {}
        if "multimodal_embedding_cache_capacity_gb" not in extra_config:
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config for DynamoMultimodalEmbeddingCacheConnector"
            )
        capacity_gb: float = extra_config["multimodal_embedding_cache_capacity_gb"]

        # --- Scheduler-side: logical LRU for CPU embedding cache ---
        # Mirrors EncoderCacheManager but for the CPU tier, tracking bytes.
        hidden_size = vllm_config.model_config.get_hidden_size()
        dtype_bytes = torch.tensor(
            [], dtype=vllm_config.model_config.dtype
        ).element_size()
        self._bytes_per_embed = hidden_size * dtype_bytes
        self._capacity_bytes = int(capacity_gb * 1024**3)

        self._cache_order: OrderedDict[str, int] = OrderedDict()  # hash → size_bytes
        self._num_used_bytes: int = 0

        self._loads_this_step: set[str] = set()
        self._saves_this_step: set[str] = set()
        self._evicts_this_step: set[str] = set()
        self._external_files: dict[str, tuple[str, int]] = {}
        bridge_dir = extra_config.get("epd_bridge_dir")
        self._bridge_dir = Path(bridge_dir) if bridge_dir else None
        self._fetch_timeout_s = float(extra_config.get("epd_fetch_timeout_s", 120.0))
        self._pending_since: dict[str, float] = {}

        write_epd_audit_event(
            "CONNECTOR_INIT",
            source_file=__file__,
            pid=os.getpid(),
            role=str(role),
            audit_dir=os.environ.get("DYN_MULTIMODAL_EPD_AUDIT_DIR"),
            bridge_dir=str(self._bridge_dir) if self._bridge_dir is not None else None,
        )

        # --- Worker-side: dumb CPU tensor store ---
        self._cpu_store: dict[str, torch.Tensor] = {}

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "capacity_gb=%.2f, capacity_bytes=%d, bytes_per_embed=%d",
            capacity_gb,
            self._capacity_bytes,
            self._bytes_per_embed,
        )

    # ==============================
    # Scheduler-side methods
    #
    # vLLM scheduler call sequence per multimodal feature:
    #
    #   1. encoder_cache_manager.check_and_update_cache(request, i)
    #      → if True (GPU hit): skip entirely, neither method below is called.
    #
    #   2. has_cache_item(identifier)
    #      → if True (CPU hit):  item goes to external_load_encoder_input
    #      → if False (CPU miss): item goes to encoder_inputs_to_schedule
    #
    #   3. update_state_after_alloc(request, i) is called for both paths.
    #      The two paths are mutually exclusive per hash within a step:
    #      - external_load_encoder_input → mm_hash IN _cache_order  → load path
    #      - encoder_inputs_to_schedule  → mm_hash NOT in _cache_order → save path
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        """Check if an embedding is in the CPU cache, promoting it to MRU on hit.

        Called by the scheduler only after the GPU encoder_cache_manager reports
        a miss. A True return tells the scheduler to skip encoder compute and
        load the embedding from the CPU store instead.
        """
        if identifier in self._cache_order:
            self._cache_order.move_to_end(identifier)
            # The UUID-specific E2E assertion relies on this diagnostic.
            logger.debug(EMBEDDING_CACHE_HIT_LOG, identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Record a load or save command for a multimodal feature.

        Called by the scheduler after has_cache_item has already determined
        the path. The _cache_order check here mirrors that decision:

        CPU hit  (mm_hash in _cache_order):  mark for CPU→GPU load.
        CPU miss (mm_hash not in _cache_order): evict LRU entries if needed,
            then mark for GPU→CPU save so the worker persists the newly
            computed embedding. Silently skips items larger than total capacity.
        """
        mm_hash: str = request.mm_features[index].identifier
        num_embeds: int = request.get_num_encoder_embeds(index)
        size_bytes: int = num_embeds * self._bytes_per_embed

        if mm_hash in self._cache_order:
            self._cache_order.move_to_end(mm_hash)
            self._loads_this_step.add(mm_hash)
            return

        if size_bytes > self._capacity_bytes:
            return

        self._saves_this_step.add(mm_hash)

        while (
            self._num_used_bytes + size_bytes > self._capacity_bytes
            and self._cache_order
        ):
            evicted_hash, evicted_bytes = self._cache_order.popitem(last=False)
            self._num_used_bytes -= evicted_bytes
            self._evicts_this_step.add(evicted_hash)

        self._cache_order[mm_hash] = size_bytes
        self._num_used_bytes += size_bytes

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        bridge_request_id = _bridge_request_id(request, self._bridge_dir)
        write_epd_audit_event(
            "CONNECTOR_ENSURE_ENTER",
            source_file=__file__,
            pid=os.getpid(),
            request_id=bridge_request_id,
            engine_request_id=request.request_id,
            num_computed_tokens=num_computed_tokens,
            num_mm_features=len(request.mm_features),
            audit_dir=os.environ.get("DYN_MULTIMODAL_EPD_AUDIT_DIR"),
            bridge_dir=str(self._bridge_dir) if self._bridge_dir is not None else None,
        )
        if self._bridge_dir is None:
            return True

        directory = request_directory(self._bridge_dir, bridge_request_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return True

        manifest = read_json(manifest_path)
        if manifest.get("request_id") != bridge_request_id:
            raise RuntimeError("vLLM EPD bridge request ID mismatch")
        generation = manifest.get("generation")
        if not isinstance(generation, str) or not generation:
            raise RuntimeError("Invalid vLLM EPD bridge generation")
        expected_identifiers = manifest.get("identifiers")
        if not isinstance(expected_identifiers, list):
            raise RuntimeError("Invalid vLLM EPD bridge manifest")
        write_epd_audit_event(
            "CONNECTOR_ENSURE",
            request_id=bridge_request_id,
            engine_request_id=request.request_id,
            generation=generation,
            num_computed_tokens=num_computed_tokens,
        )

        uncovered: list[tuple[int, str, int]] = []
        kv_covered = 0
        p_cache_covered = 0
        for index, feature in enumerate(request.mm_features):
            end = feature.mm_position.offset + feature.mm_position.length
            if end <= num_computed_tokens:
                kv_covered += 1
                continue
            identifier = feature.identifier
            if (
                index >= len(expected_identifiers)
                or expected_identifiers[index] != identifier
            ):
                raise RuntimeError("vLLM EPD bridge feature identity mismatch")
            if identifier in self._cache_order:
                p_cache_covered += 1
                continue
            uncovered.append((index, identifier, request.get_num_encoder_embeds(index)))

        if not uncovered:
            logger.info(
                "vLLM post-KV EPD no fetch: request_id=%s kv_covered=%d "
                "p_cache_covered=%d",
                bridge_request_id,
                kv_covered,
                p_cache_covered,
            )
            write_epd_audit_event(
                "CONNECTOR_NO_FETCH",
                request_id=bridge_request_id,
                engine_request_id=request.request_id,
                generation=generation,
                kv_covered=kv_covered,
                p_cache_covered=p_cache_covered,
            )
            self._pending_since.pop(bridge_request_id, None)
            return True

        error_path = directory / "error.json"
        if error_path.exists():
            error = read_json(error_path)
            if error.get("generation") == generation:
                raise RuntimeError(
                    "vLLM post-KV embedding fetch failed: "
                    f"{error.get('message', 'unknown error')}"
                )
            error_path.unlink(missing_ok=True)

        ready_path = directory / "ready.json"
        if ready_path.exists():
            ready = read_json(ready_path)
            if ready.get("generation") != generation:
                ready_path.unlink(missing_ok=True)
            else:
                ready_by_identifier = {
                    str(item["identifier"]): item for item in ready.get("items", [])
                }
                for index, identifier, num_embeds in uncovered:
                    item = ready_by_identifier.get(identifier)
                    if item is None or int(item.get("index", -1)) != index:
                        raise RuntimeError(
                            f"vLLM post-KV embedding response is incomplete for {identifier}"
                        )
                    item_path = Path(str(item["path"]))
                    if not item_path.is_file():
                        raise RuntimeError(
                            f"vLLM post-KV embedding file disappeared for {identifier}"
                        )
                    self._reserve_external_item(identifier, num_embeds, item_path)
                write_epd_audit_event(
                    "CONNECTOR_READY",
                    request_id=bridge_request_id,
                    engine_request_id=request.request_id,
                    generation=generation,
                    indices=[index for index, _identifier, _num_embeds in uncovered],
                )
                self._pending_since.pop(bridge_request_id, None)
                return True

        need_path = directory / "need.json"
        if not need_path.exists():
            atomic_write_json(
                need_path,
                {
                    "request_id": bridge_request_id,
                    "generation": generation,
                    "items": [
                        {"index": index, "identifier": identifier}
                        for index, identifier, _num_embeds in uncovered
                    ],
                },
            )
            logger.info(
                "vLLM post-KV EPD fetch requested: request_id=%s generation=%s "
                "indices=%s kv_covered=%d p_cache_covered=%d",
                bridge_request_id,
                generation,
                [index for index, _identifier, _num_embeds in uncovered],
                kv_covered,
                p_cache_covered,
            )
            write_epd_audit_event(
                "CONNECTOR_FETCH_REQUESTED",
                request_id=bridge_request_id,
                engine_request_id=request.request_id,
                generation=generation,
                indices=[index for index, _identifier, _num_embeds in uncovered],
                kv_covered=kv_covered,
                p_cache_covered=p_cache_covered,
            )
            self._pending_since[bridge_request_id] = time.monotonic()

        started = self._pending_since.setdefault(bridge_request_id, time.monotonic())
        if time.monotonic() - started > self._fetch_timeout_s:
            raise TimeoutError(
                f"Timed out waiting for post-KV embeddings for {bridge_request_id}"
            )
        return False

    def _reserve_external_item(
        self, identifier: str, num_embeds: int, item_path: Path
    ) -> None:
        size_bytes = num_embeds * self._bytes_per_embed
        if size_bytes > self._capacity_bytes:
            raise RuntimeError(
                f"Remote embedding {identifier} exceeds configured P cache capacity"
            )
        while (
            self._num_used_bytes + size_bytes > self._capacity_bytes
            and self._cache_order
        ):
            evicted_hash, evicted_bytes = self._cache_order.popitem(last=False)
            self._num_used_bytes -= evicted_bytes
            self._evicts_this_step.add(evicted_hash)
            self._external_files.pop(evicted_hash, None)
        self._cache_order[identifier] = size_bytes
        self._num_used_bytes += size_bytes
        self._external_files[identifier] = (str(item_path), num_embeds)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush accumulated load/save/evict commands into metadata for the worker."""
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=list(self._loads_this_step),
            saves=list(self._saves_this_step),
            evicts=list(self._evicts_this_step),
            external_loads={
                identifier: self._external_files[identifier]
                for identifier in self._loads_this_step
                if identifier in self._external_files
            },
        )

        self._loads_this_step.clear()
        self._saves_this_step.clear()
        self._evicts_this_step.clear()
        return meta

    # ==============================
    # Worker-side methods
    #
    # Called by the model runner each step with the metadata produced by
    # build_connector_meta. The worker has no caching logic of its own;
    # it simply obeys the scheduler's load/save/evict commands.
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Copy cached embeddings from CPU store to GPU encoder_cache, and evict
        entries the scheduler marked for removal.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        for mm_hash in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            if mm_hash not in self._cpu_store:
                external = metadata.external_loads.get(mm_hash)
                if external is None:
                    raise RuntimeError(
                        f"start_load_caches: hash {mm_hash} missing from cpu_store"
                    )
                path, num_embeds = external
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"External embedding {mm_hash} is not a tensor")
                if (
                    tensor.ndim >= 2
                    and tensor.shape[0] == 1
                    and tensor.shape[1] == num_embeds
                ):
                    tensor = tensor.squeeze(0)
                if tensor.ndim == 0 or tensor.shape[0] != num_embeds:
                    raise RuntimeError(
                        f"External embedding {mm_hash} shape {tuple(tensor.shape)} "
                        f"does not match {num_embeds} encoder tokens"
                    )
                self._cpu_store[mm_hash] = tensor.contiguous()
                logger.info(
                    "vLLM post-KV EPD worker loaded external embedding: "
                    "identifier=%s path=%s",
                    mm_hash,
                    path,
                )
                write_epd_audit_event(
                    "WORKER_LOADED_EXTERNAL",
                    identifier=mm_hash,
                    path=path,
                )
                if self._bridge_dir is not None:
                    write_cache_event(self._bridge_dir, "ADD", mm_hash)
            encoder_cache[mm_hash] = self._cpu_store[mm_hash].to(
                self._device, non_blocking=True
            )

        for mm_hash in metadata.evicts:
            removed = self._cpu_store.pop(mm_hash, None)
            if removed is not None and self._bridge_dir is not None:
                write_cache_event(self._bridge_dir, "REMOVE", mm_hash)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Copy a newly computed embedding from GPU encoder_cache to CPU store."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        if mm_hash not in metadata.saves:
            return
        if mm_hash in self._cpu_store:
            return
        if mm_hash not in encoder_cache:
            logger.warning(
                "save_caches: hash %s in metadata.saves but not in encoder_cache",
                mm_hash,
            )
            return
        self._cpu_store[mm_hash] = encoder_cache[mm_hash].cpu()
        if self._bridge_dir is not None:
            write_cache_event(self._bridge_dir, "ADD", mm_hash)

    def request_finished(self, request: "Request") -> tuple[bool, None]:
        self._pending_since.pop(_bridge_request_id(request), None)
        return False, None

    def shutdown(self) -> None:
        if self._bridge_dir is not None:
            for mm_hash in self._cpu_store:
                write_cache_event(self._bridge_dir, "REMOVE", mm_hash)
        self._cpu_store.clear()
