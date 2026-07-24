# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from dynamo.llm import MultimodalEmbeddingCachePublisher

    from .prefill_worker_utils import MultiModalEmbeddingLoader

logger = logging.getLogger(__name__)


def request_directory(root: str | Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return Path(root) / "requests" / digest


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_cache_event(root: str | Path, action: str, cache_key: str) -> None:
    event_path = Path(root) / "events" / f"{uuid.uuid4().hex}.json"
    atomic_write_json(event_path, {"action": action, "cache_key": cache_key})


def write_epd_audit_event(event: str, **fields: Any) -> None:
    audit_dir = os.environ.get("DYN_MULTIMODAL_EPD_AUDIT_DIR")
    if not audit_dir:
        return
    timestamp_ns = time.time_ns()
    event_path = Path(audit_dir) / f"{timestamp_ns}-{uuid.uuid4().hex}.json"
    try:
        atomic_write_json(
            event_path,
            {"event": event, "timestamp_ns": timestamp_ns, **fields},
        )
    except OSError:
        logger.warning("Failed to write multimodal EPD audit event", exc_info=True)


@dataclass(frozen=True)
class _RegisteredRequest:
    request_id: str
    generation: str
    image_urls: tuple[str, ...]
    identifiers: tuple[str, ...]
    model: str
    routing_plan: dict[str, Any] | None
    context: Any


class VllmEpdEmbeddingBridge:
    """Bridge parent-side Dynamo E fetches into vLLM's EngineCore process."""

    def __init__(
        self,
        root: str | Path,
        embedding_loader: "MultiModalEmbeddingLoader",
        cache_publisher: "MultimodalEmbeddingCachePublisher | None" = None,
        *,
        poll_interval_s: float = 0.01,
    ) -> None:
        self.root = Path(root)
        self.embedding_loader = embedding_loader
        self.cache_publisher = cache_publisher
        self.poll_interval_s = poll_interval_s
        self._requests: dict[str, _RegisteredRequest] = {}
        self._fetch_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        (self.root / "requests").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        self._poll_task = asyncio.create_task(self._run())

    async def register(
        self,
        *,
        request_id: str,
        image_urls: list[str],
        identifiers: list[str],
        model: str,
        routing_plan: dict[str, Any] | None,
        context: Any,
    ) -> None:
        if len(image_urls) != len(identifiers) or not image_urls:
            raise ValueError(
                "EPD bridge URLs and identifiers must be non-empty and aligned"
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("EPD bridge identifiers must be unique within one request")

        await self.unregister(request_id)
        directory = request_directory(self.root, request_id)
        directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        registration = _RegisteredRequest(
            request_id=request_id,
            generation=generation,
            image_urls=tuple(image_urls),
            identifiers=tuple(identifiers),
            model=model,
            routing_plan=routing_plan,
            context=context,
        )
        self._requests[directory.name] = registration
        atomic_write_json(
            directory / "manifest.json",
            {
                "request_id": request_id,
                "generation": generation,
                "identifiers": identifiers,
            },
        )
        write_epd_audit_event(
            "BRIDGE_REGISTER",
            request_id=request_id,
            generation=generation,
            identifiers=identifiers,
        )

    async def unregister(self, request_id: str) -> None:
        directory = request_directory(self.root, request_id)
        self._requests.pop(directory.name, None)
        task = self._fetch_tasks.pop(directory.name, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        shutil.rmtree(directory, ignore_errors=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._poll_task.cancel()
        try:
            await self._poll_task
        except asyncio.CancelledError:
            pass
        tasks = list(self._fetch_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def cancel(self) -> None:
        self._closed = True
        self._poll_task.cancel()
        for task in self._fetch_tasks.values():
            task.cancel()

    async def _run(self) -> None:
        while True:
            await self._start_pending_fetches()
            self._publish_cache_events()
            await asyncio.sleep(self.poll_interval_s)

    async def _start_pending_fetches(self) -> None:
        for need_path in (self.root / "requests").glob("*/need.json"):
            request_key = need_path.parent.name
            existing = self._fetch_tasks.get(request_key)
            if existing is not None and not existing.done():
                continue
            task = asyncio.create_task(self._process_need(need_path))
            self._fetch_tasks[request_key] = task

    async def _process_need(self, need_path: Path) -> None:
        directory = need_path.parent
        registration = self._requests.get(directory.name)
        try:
            if registration is None:
                raise RuntimeError("EPD bridge request is no longer registered")
            need = read_json(need_path)
            if need.get("request_id") != registration.request_id:
                raise ValueError("EPD bridge request ID mismatch")
            if need.get("generation") != registration.generation:
                raise ValueError("EPD bridge request generation mismatch")
            raw_items = need.get("items")
            if not isinstance(raw_items, list) or not raw_items:
                raise ValueError("EPD bridge need file contains no items")

            indices: list[int] = []
            identifiers: list[str] = []
            for item in raw_items:
                index = int(item["index"])
                identifier = str(item["identifier"])
                if index < 0 or index >= len(registration.image_urls):
                    raise ValueError("EPD bridge object index is out of range")
                if registration.identifiers[index] != identifier:
                    raise ValueError("EPD bridge object identity mismatch")
                indices.append(index)
                identifiers.append(identifier)
            if len(set(indices)) != len(indices):
                raise ValueError("EPD bridge object indices must be unique")

            routing_plan = self._subset_routing_plan(registration.routing_plan, indices)
            logger.info(
                "vLLM post-KV EPD dispatch: request_id=%s generation=%s indices=%s",
                registration.request_id,
                registration.generation,
                indices,
            )
            write_epd_audit_event(
                "BRIDGE_DISPATCH",
                request_id=registration.request_id,
                generation=registration.generation,
                indices=indices,
            )
            parts = await self.embedding_loader.load_multimodal_embedding_parts(
                [registration.image_urls[index] for index in indices],
                registration.request_id,
                model=registration.model,
                routing_plan=routing_plan,
                context=registration.context,
            )
            if len(parts) != len(indices):
                raise ValueError("EPD bridge received an incomplete embedding response")

            ready_items = []
            for index, identifier, part in zip(
                indices, identifiers, parts, strict=True
            ):
                tensor = part.tensor.detach().cpu().contiguous()
                item_path = directory / f"item-{index}-{registration.generation}.pt"
                temporary = directory / f".item-{index}.{uuid.uuid4().hex}.tmp"
                torch.save(tensor, temporary)
                os.replace(temporary, item_path)
                ready_items.append(
                    {
                        "index": index,
                        "identifier": identifier,
                        "path": str(item_path),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )
            atomic_write_json(
                directory / "ready.json",
                {
                    "request_id": registration.request_id,
                    "generation": registration.generation,
                    "items": ready_items,
                },
            )
            logger.info(
                "vLLM post-KV EPD ready: request_id=%s generation=%s indices=%s",
                registration.request_id,
                registration.generation,
                indices,
            )
            write_epd_audit_event(
                "BRIDGE_READY",
                request_id=registration.request_id,
                generation=registration.generation,
                indices=indices,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Failed to fulfill vLLM post-KV embedding request")
            atomic_write_json(
                directory / "error.json",
                {
                    "request_id": (
                        registration.request_id if registration is not None else None
                    ),
                    "generation": (
                        registration.generation if registration is not None else None
                    ),
                    "message": str(error),
                },
            )
        finally:
            need_path.unlink(missing_ok=True)

    @staticmethod
    def _subset_routing_plan(
        routing_plan: dict[str, Any] | None, indices: list[int]
    ) -> dict[str, Any] | None:
        if routing_plan is None:
            return None
        raw_objects = routing_plan.get("objects")
        if not isinstance(raw_objects, list):
            return None
        selected = []
        for new_index, original_index in enumerate(indices):
            if original_index >= len(raw_objects):
                return None
            raw = raw_objects[original_index]
            if not isinstance(raw, dict):
                return None
            selected.append({**raw, "object_index": new_index})
        return {**routing_plan, "objects": selected}

    def _publish_cache_events(self) -> None:
        if self.cache_publisher is None:
            return
        for event_path in (self.root / "events").glob("*.json"):
            try:
                event = read_json(event_path)
                cache_key = str(event["cache_key"])
                action = event["action"]
                if action == "ADD":
                    self.cache_publisher.publish_delta([cache_key], [])
                elif action == "REMOVE":
                    self.cache_publisher.publish_delta([], [cache_key])
                else:
                    raise ValueError(f"Unknown cache event action: {action}")
                event_path.unlink(missing_ok=True)
            except Exception:
                logger.warning(
                    "Failed to publish vLLM engine embedding-cache event",
                    exc_info=True,
                )
