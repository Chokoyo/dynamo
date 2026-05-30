# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

MINIMUM_VLLM_VERSION = "0.17.0"

logger = logging.getLogger(__name__)

# Round-3 instrumentation (exp-3): the scheduler-authoritative connector lives
# inside the vLLM EngineCore subprocess so its in-memory counters are invisible
# to the Dynamo Prometheus callback (which runs in the API-server process).
# Bridge the two with an atomically-written JSON snapshot at a well-known path.
# The Prometheus scrape callback (registered in worker_factory.py) reads the
# same file. See exp-3 RESULTS.md §3 for the gap this closes.
_EC_CONNECTOR_STATS_ENV = "DYN_EC_CONNECTOR_STATS_PATH"


def _default_stats_path() -> str:
    return os.path.join(tempfile.gettempdir(), "dyn_ec_connector_stats.json")


def resolve_ec_connector_stats_path() -> str:
    """Return the path the connector writes its stats snapshot to.

    Honors ``$DYN_EC_CONNECTOR_STATS_PATH`` so the launcher can isolate
    per-job state. Also consumed by ``register_ec_connector_metrics``.
    """
    return os.environ.get(_EC_CONNECTOR_STATS_ENV, _default_stats_path())


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)


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

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError(
                "ec_transfer_config must be set for DynamoMultimodalEmbeddingCacheConnector"
            )

        if "multimodal_embedding_cache_capacity_gb" not in (
            transfer_config.ec_connector_extra_config or {}
        ):
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config for DynamoMultimodalEmbeddingCacheConnector"
            )
        capacity_gb: float = transfer_config.ec_connector_extra_config[
            "multimodal_embedding_cache_capacity_gb"
        ]

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

        # --- Worker-side: dumb CPU tensor store ---
        self._cpu_store: dict[str, torch.Tensor] = {}

        # --- Round-3 instrumentation: monotonic stats + JSON snapshot file ---
        # All counters are monotonic; the Prometheus side computes deltas.
        # "hits" = CPU cache hits (scheduler skips encoder, loads from CPU store).
        # "misses" = CPU cache misses (scheduler dispatches the encoder).
        # "evictions" = entries pushed out of the LRU to make room.
        self._stats_lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "lookups": 0,
        }
        self._stats_path = resolve_ec_connector_stats_path()
        self._capacity_gb = capacity_gb
        # Buffer of events flushed alongside the stats snapshot (patch 2).
        # See _emit_event(): one event per save and one per evict, dropped after
        # being read by the scrape callback. Capped to avoid unbounded growth
        # if the consumer is wedged.
        self._event_buf_max = 2048
        self._event_buf: list[dict] = []
        # Robust against unit tests that pass a MagicMock for transfer_config.
        _eid = getattr(transfer_config, "engine_id", "")
        self._engine_id = _eid if isinstance(_eid, str) else ""
        # Write an initial all-zeros snapshot so the consumer never sees the
        # file as missing right after startup.
        self._flush_stats_snapshot()

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "capacity_gb=%.2f, capacity_bytes=%d, bytes_per_embed=%d, "
            "stats_path=%s engine_id=%s",
            capacity_gb,
            self._capacity_bytes,
            self._bytes_per_embed,
            self._stats_path,
            self._engine_id,
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

    # ---- Round-3 instrumentation helpers (internal) ----

    def _emit_event(self, kind: str, mm_hash: str, size_bytes: int = 0) -> None:
        """Patch-2 stretch: buffer an EmbeddingEvent for the event-plane plumbing.

        Format intentionally mirrors KV-Events: a flat dict with a discriminator
        plus a few small fields. The buffer is drained on the next stats flush.
        """
        ev = {
            "kind": kind,
            "mm_hash": mm_hash,
            "size_bytes": size_bytes,
            "engine_id": self._engine_id,
            "ts": time.time(),
        }
        if len(self._event_buf) < self._event_buf_max:
            self._event_buf.append(ev)

    def _flush_stats_snapshot(self) -> None:
        """Atomically write the current stats + buffered events to disk.

        Format::
            {
              "stats": {"hits": int, "misses": int, "evictions": int, ...},
              "gauges": {"entries": int, "current_bytes": int,
                         "capacity_bytes": int, "utilization": float},
              "events": [ {kind, mm_hash, size_bytes, engine_id, ts}, ... ],
              "ts": float
            }

        Atomicity: write to a temp sibling then ``os.replace`` so the consumer
        never reads a half-written file.
        """
        with self._stats_lock:
            entries = len(self._cache_order)
            used = self._num_used_bytes
            util = (
                (used / self._capacity_bytes) if self._capacity_bytes > 0 else 0.0
            )
            payload = {
                "stats": dict(self._stats),
                "gauges": {
                    "entries": entries,
                    "current_bytes": used,
                    "capacity_bytes": self._capacity_bytes,
                    "utilization": util,
                },
                "events": list(self._event_buf),
                "ts": time.time(),
                "engine_id": self._engine_id,
            }
            self._event_buf.clear()

        try:
            tmp = f"{self._stats_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._stats_path)
        except Exception as exc:
            # Never let metric I/O break the scheduler.
            logger.debug("ec-connector stats flush failed: %s", exc)

    def has_cache_item(self, identifier: str) -> bool:
        """Check if an embedding is in the CPU cache, promoting it to MRU on hit.

        Called by the scheduler only after the GPU encoder_cache_manager reports
        a miss. A True return tells the scheduler to skip encoder compute and
        load the embedding from the CPU store instead.
        """
        with self._stats_lock:
            self._stats["lookups"] += 1
            if identifier in self._cache_order:
                self._stats["hits"] += 1
                self._cache_order.move_to_end(identifier)
                return True
            self._stats["misses"] += 1
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
            with self._stats_lock:
                self._stats["evictions"] += 1
            self._emit_event("evict", evicted_hash, evicted_bytes)

        self._cache_order[mm_hash] = size_bytes
        self._num_used_bytes += size_bytes
        self._emit_event("save", mm_hash, size_bytes)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush accumulated load/save/evict commands into metadata for the worker."""
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=list(self._loads_this_step),
            saves=list(self._saves_this_step),
            evicts=list(self._evicts_this_step),
        )

        self._loads_this_step.clear()
        self._saves_this_step.clear()
        self._evicts_this_step.clear()

        # Round-3 instrumentation: persist the latest stats snapshot so the
        # Prometheus scrape callback (worker_factory.py) can read it.
        # Cheap (<1 ms): single small JSON file, one fsync-free os.replace.
        self._flush_stats_snapshot()
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
            if mm_hash in self._cpu_store:
                encoder_cache[mm_hash] = self._cpu_store[mm_hash].to(
                    "cuda", non_blocking=True
                )
            else:
                logger.warning(
                    "start_load_caches: hash %s not in cpu_store, skipping", mm_hash
                )

        for mm_hash in metadata.evicts:
            self._cpu_store.pop(mm_hash, None)

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
