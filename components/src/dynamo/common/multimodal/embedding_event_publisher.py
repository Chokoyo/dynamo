# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ PUB-side event publisher for ``DynamoMultimodalEmbeddingCacheConnector``.

The connector (which lives inside the vLLM EngineCore subprocess) calls
:func:`EmbeddingEventPublisher.publish` whenever it issues a save or evict
on the CPU embedding cache. Consumers (e.g. a cross-node router or a
secondary replica that wants to populate its own connector tier
opportunistically) can subscribe to the PUB socket via the matching
:class:`EmbeddingEventSubscriber` helper.

Design constraints
------------------
* **Optional and additive.** The publisher is OFF unless
  ``$DYN_EC_EVENT_PUB_ENDPOINT`` is set in the environment of the EngineCore
  subprocess. The existing JSON-snapshot bridge (R3) remains the metrics
  path; ZMQ is the *cross-process* dissemination path that R5 will need.
* **Non-blocking, drop-on-full.** A wedged consumer must never stall the
  scheduler. We use ``zmq.NOBLOCK`` + a ZMQ_SNDHWM of 1024; overflow is
  counted in ``self.dropped`` and logged at WARN at most once per 100
  drops.
* **Fork-aware.** ZMQ contexts may not be shared across forks. The
  publisher lazily creates its context+socket on the first
  :func:`publish` call so it gets bound inside the EngineCore subprocess,
  not in the parent.
* **Wire format.** msgpack if available, JSON otherwise — same dict
  schema as the file-bridge ``events`` list: ``{kind, mm_hash, size_bytes,
  engine_id, ts}``.

Mirrors what ``dynamo.llm.KvEventPublisher`` does for KV-cache blocks; the
KV variant is a Rust binding for performance and uses CBOR. Our Python
version is small enough to ship as a stretch goal for exp-3 R4 without
touching the Rust crate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# Env var read by both the connector (publisher side) and any subscriber
# helper. Example: ``tcp://0.0.0.0:5557`` for the publisher,
# ``tcp://localhost:5557`` for a subscriber.
_PUB_ENDPOINT_ENV = "DYN_EC_EVENT_PUB_ENDPOINT"
_TOPIC_ENV = "DYN_EC_EVENT_TOPIC"
_DEFAULT_TOPIC = b"ec_embedding_event"
_DEFAULT_HWM = 1024


def resolve_ec_event_pub_endpoint() -> Optional[str]:
    """Return the configured PUB endpoint, or ``None`` if disabled."""
    return os.environ.get(_PUB_ENDPOINT_ENV) or None


def resolve_ec_event_topic() -> bytes:
    """Return the topic prefix used on the PUB/SUB wire."""
    topic = os.environ.get(_TOPIC_ENV)
    if topic:
        return topic.encode("utf-8")
    return _DEFAULT_TOPIC


def _serialize(payload: dict) -> bytes:
    """Pick msgpack if available, else JSON. Subscribers must mirror this."""
    try:
        import msgpack  # type: ignore[import-not-found]

        return msgpack.packb(payload, use_bin_type=True)
    except Exception:
        return json.dumps(payload).encode("utf-8")


def _deserialize(buf: bytes) -> dict:
    try:
        import msgpack  # type: ignore[import-not-found]

        return msgpack.unpackb(buf, raw=False)
    except Exception:
        return json.loads(buf.decode("utf-8"))


class EmbeddingEventPublisher:
    """Optional ZMQ PUB sender for embedding cache events.

    Construction is cheap and does NOT bind the socket. The first
    :func:`publish` call binds (so the context lives inside the calling
    process). If ``$DYN_EC_EVENT_PUB_ENDPOINT`` is unset, the publisher
    becomes a no-op.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        topic: Optional[bytes] = None,
        hwm: int = _DEFAULT_HWM,
        engine_id: str = "",
    ) -> None:
        self._endpoint = endpoint or resolve_ec_event_pub_endpoint()
        self._topic = topic or resolve_ec_event_topic()
        self._hwm = hwm
        self._engine_id = engine_id
        self._ctx = None  # type: ignore[var-annotated]
        self._socket = None  # type: ignore[var-annotated]
        self._lock = threading.Lock()
        # Public counters; the connector mirrors these into its stats dict.
        self.published: int = 0
        self.dropped: int = 0
        self._init_failed = False

    @property
    def enabled(self) -> bool:
        return bool(self._endpoint) and not self._init_failed

    def _ensure_socket(self) -> bool:
        """Lazy-init the ZMQ context+socket. Returns False on failure."""
        if self._init_failed:
            return False
        if self._socket is not None:
            return True
        try:
            import zmq  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning(
                "EmbeddingEventPublisher: pyzmq not importable; disabling (%s)",
                exc,
            )
            self._init_failed = True
            return False
        try:
            self._ctx = zmq.Context.instance()
            self._socket = self._ctx.socket(zmq.PUB)
            self._socket.setsockopt(zmq.SNDHWM, self._hwm)
            # Linger=0 so we never block on process shutdown even if a peer
            # has buffered un-acked messages.
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(self._endpoint)
            logger.info(
                "EmbeddingEventPublisher bound: endpoint=%s topic=%s hwm=%d engine_id=%s",
                self._endpoint,
                self._topic,
                self._hwm,
                self._engine_id,
            )
        except Exception as exc:
            logger.warning(
                "EmbeddingEventPublisher bind failed (%s) — disabling.", exc
            )
            self._init_failed = True
            try:
                if self._socket is not None:
                    self._socket.close()
            except Exception:
                pass
            self._socket = None
            return False
        return True

    def publish(self, event: dict) -> None:
        """Send one ``EmbeddingEvent`` dict. Non-blocking; drops on overflow.

        Safe to call from the EngineCore step thread — the call does at most
        one non-blocking ZMQ ``send_multipart``.
        """
        if not self._endpoint:
            return
        with self._lock:
            if not self._ensure_socket():
                return
            try:
                import zmq  # type: ignore[import-not-found]

                ts = event.get("ts") or time.time()
                payload = dict(event)
                payload["ts"] = ts
                if "engine_id" not in payload:
                    payload["engine_id"] = self._engine_id
                buf = _serialize(payload)
                try:
                    self._socket.send_multipart(
                        [self._topic, buf], flags=zmq.NOBLOCK
                    )
                    self.published += 1
                except zmq.Again:
                    self.dropped += 1
                    if self.dropped % 100 == 1:
                        logger.warning(
                            "EmbeddingEventPublisher: HWM full, dropped=%d",
                            self.dropped,
                        )
            except Exception as exc:
                # Never let event publishing break the scheduler.
                logger.debug("EmbeddingEventPublisher.publish failed: %s", exc)

    def close(self) -> None:
        with self._lock:
            try:
                if self._socket is not None:
                    self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
            # Do not term the shared context; other components may use it.


class EmbeddingEventSubscriber:
    """Helper used by tests and downstream routers to consume EmbeddingEvents.

    Not used by the connector itself. Provided so cross-node R5 work has a
    documented receiving side.

    Example::

        sub = EmbeddingEventSubscriber("tcp://localhost:5557")
        for ev in sub.iter_events(max_events=10, timeout_ms=1000):
            print(ev)
    """

    def __init__(
        self,
        endpoint: str,
        topic: Optional[bytes] = None,
    ) -> None:
        import zmq  # type: ignore[import-not-found]

        self._endpoint = endpoint
        self._topic = topic or resolve_ec_event_topic()
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(endpoint)

    def iter_events(self, max_events: int = -1, timeout_ms: int = 1000):
        import zmq  # type: ignore[import-not-found]

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        count = 0
        deadline = time.time() + (timeout_ms / 1000.0)
        while max_events < 0 or count < max_events:
            remaining = max(0, int((deadline - time.time()) * 1000))
            socks = dict(poller.poll(remaining))
            if self._socket not in socks:
                break
            frames = self._socket.recv_multipart()
            if len(frames) < 2:
                continue
            ev = _deserialize(frames[1])
            yield ev
            count += 1

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass


__all__ = [
    "EmbeddingEventPublisher",
    "EmbeddingEventSubscriber",
    "resolve_ec_event_pub_endpoint",
    "resolve_ec_event_topic",
]
