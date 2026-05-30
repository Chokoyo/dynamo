# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import TYPE_CHECKING, Optional

from dynamo._core import VirtualConnectorCoordinator
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.mdc import MdcEntry, select_entry, worker_info_from_mdc
from dynamo.planner.errors import EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

if TYPE_CHECKING:
    from dynamo.llm import FpmEventSubscriber

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Constants for scaling readiness check and waiting
SCALING_CHECK_INTERVAL = int(
    os.environ.get("SCALING_CHECK_INTERVAL", 10)
)  # Check every 10 seconds
SCALING_MAX_WAIT_TIME = int(
    os.environ.get("SCALING_MAX_WAIT_TIME", 1800)
)  # Maximum wait time: 30 minutes (1800 seconds)
SCALING_MAX_RETRIES = SCALING_MAX_WAIT_TIME // SCALING_CHECK_INTERVAL  # 180 retries


def _mdc_entries_from_subscriber(
    subscriber: Optional["FpmEventSubscriber"],
) -> list[MdcEntry]:
    """Read the discovery-captured card JSON snapshot from an FPM subscriber.

    Returns an empty list if tracking has not been started yet or no cards
    have been observed.  Discovery-sourced entries have no wrapper
    component/endpoint (those come from the CRD in K8s mode); worker_info_from_mdc
    falls back to backend defaults for those fields.
    """
    if subscriber is None:
        return []
    try:
        cards = subscriber.get_model_cards()
    except RuntimeError:
        # start_tracking() not called yet.
        return []

    entries: list[MdcEntry] = []
    for worker_id, card_str in cards.items():
        try:
            card_json = json.loads(card_str)
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed MDC card JSON for worker {worker_id}")
            continue
        entries.append(MdcEntry(card_json=card_json, instance_id=worker_id))
    return entries


class VirtualConnector(PlannerConnector):
    """
    This is a virtual connector for planner to output scaling decisions to non-native environments
    This virtual connector does not actually scale the deployment, instead, it communicates with the non-native environment through dynamo-runtime's VirtualConnectorCoordinator.
    The deployment environment needs to use VirtualConnectorClient (in the Rust/Python bindings) to read from the scaling decisions and update report scaling status.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        dynamo_namespace: str,
        model_name: Optional[str] = None,
    ):
        self.connector = VirtualConnectorCoordinator(
            runtime,
            dynamo_namespace,
            SCALING_CHECK_INTERVAL,
            SCALING_MAX_WAIT_TIME,
            SCALING_MAX_RETRIES,
        )

        if model_name is None:
            raise ValueError("Model name is required for virtual connector")

        self.model_name = model_name.lower()  # normalize model name to lowercase (MDC)

        self.dynamo_namespace = dynamo_namespace

        # MDC sources injected by NativePlannerBase after FPM subscribers exist.
        self._prefill_mdc_sub: Optional["FpmEventSubscriber"] = None
        self._decode_mdc_sub: Optional["FpmEventSubscriber"] = None

        # exp-4 R6: Python-side encoder replica tracking. The Rust
        # binding does not yet take num_encode (only num_prefill /
        # num_decode). We mirror the routing here so the planner can
        # emit ENCODER targets and downstream consumers (deployment
        # orchestrators / smoke tests) can read the desired count.
        self._desired_num_encode: Optional[int] = None
        # Optional callback the deployment glue can register to
        # actually act on the encode-replica decision. Signature:
        #   async def on_encode_replica_decision(new: int, prev: Optional[int]) -> None
        # If None the decision is recorded + logged only.
        self._encode_decision_callback = None
        # Optional subscriber for encoder MDC entries (mirrors prefill / decode).
        self._encode_mdc_sub: Optional["FpmEventSubscriber"] = None

    def set_mdc_subscribers(
        self,
        prefill: Optional["FpmEventSubscriber"] = None,
        decode: Optional["FpmEventSubscriber"] = None,
        encode: Optional["FpmEventSubscriber"] = None,
    ) -> None:
        """Inject FPM subscribers used as the MDC source for get_worker_info.

        VirtualConnector has no K8s CRDs to read, so it reads model cards
        from the discovery watch maintained by the FPM subscribers.  Until
        this is called, get_worker_info returns backend defaults only.
        """
        self._prefill_mdc_sub = prefill
        self._decode_mdc_sub = decode
        self._encode_mdc_sub = encode

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """Populate WorkerInfo from discovery-sourced MDCs, with defaults fallback.

        Called by ``resolve_worker_info`` (once, at init) and by the tick-loop
        refresh (once cards are available in discovery).
        """
        if sub_component_type == SubComponentType.PREFILL:
            subscriber = self._prefill_mdc_sub
        elif sub_component_type == SubComponentType.ENCODER:
            # exp-4 R6: encoder MDC source. Defaults to None until
            # the deployment glue calls set_mdc_subscribers(encode=).
            subscriber = self._encode_mdc_sub
        else:
            subscriber = self._decode_mdc_sub
        entries = _mdc_entries_from_subscriber(subscriber)
        entry = select_entry(entries, sub_component_type)
        if entry is not None:
            info = worker_info_from_mdc(
                entry,
                sub_component_type,
                backend=backend,
            )
            if not info.model_name:
                info.model_name = self.model_name
            return info

        info = build_worker_info_from_defaults(backend, sub_component_type)
        info.model_name = self.model_name
        return info

    async def _async_init(self):
        """Async initialization that must be called after __init__"""
        await self.connector.async_init()

    async def _update_scaling_decision(
        self, num_prefill: Optional[int] = None, num_decode: Optional[int] = None
    ):
        """Update scaling decision"""
        await self.connector.update_scaling_decision(num_prefill, num_decode)

    async def _update_encode_decision(self, num_encode: int) -> None:
        """exp-4 R6: record + fan out an encoder-replica decision.

        Mirrors the prefill / decode path but lives entirely in Python
        because the Rust ``update_scaling_decision`` signature does not
        yet take ``num_encode``. If a deployment-level callback was
        registered via :py:meth:`set_encode_decision_callback`, it is
        invoked with ``(new, prev)``. The latest decision is stored on
        ``self._desired_num_encode`` and exposed via
        :py:meth:`get_desired_encode_replicas` so observers can poll.
        """
        prev = self._desired_num_encode
        if prev == num_encode:
            logger.debug(
                f"[VirtualConnector] encode replicas already at {num_encode}, no-op"
            )
            return
        logger.info(
            f"[VirtualConnector] encode replica decision: {prev} -> {num_encode}"
        )
        self._desired_num_encode = num_encode
        cb = self._encode_decision_callback
        if cb is None:
            return
        try:
            res = cb(num_encode, prev)
            # Support both sync and async callbacks.
            if res is not None and hasattr(res, "__await__"):
                await res
        except Exception as e:
            logger.warning(
                f"[VirtualConnector] encode_decision_callback raised: {e}"
            )

    def set_encode_decision_callback(self, cb) -> None:
        """Register a callback fired whenever the encoder replica target changes.

        Signature: ``cb(new: int, prev: Optional[int]) -> None | Awaitable[None]``.
        Used by deployment substrates (smoke harness, K8s glue) that
        want to actually act on the encoder decision until the Rust
        binding learns about ``num_encode``.
        """
        self._encode_decision_callback = cb

    def get_desired_encode_replicas(self) -> Optional[int]:
        """Read the latest encoder replica target (None if never set)."""
        return self._desired_num_encode

    async def _wait_for_scaling_completion(self):
        """Wait for the deployment environment to report that scaling is complete"""
        await self.connector.wait_for_scaling_completion()

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Add a component by increasing its replica count by 1"""
        state = self.connector.read_state()

        if sub_component_type == SubComponentType.PREFILL:
            await self._update_scaling_decision(
                num_prefill=state.num_prefill_workers + 1
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._update_scaling_decision(num_decode=state.num_decode_workers + 1)
        elif sub_component_type == SubComponentType.ENCODER:
            # exp-4 R6: encoder replicas tracked Python-side.
            current = self._desired_num_encode if self._desired_num_encode is not None else 0
            await self._update_encode_decision(current + 1)

        if blocking:
            await self._wait_for_scaling_completion()

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""
        state = self.connector.read_state()

        if sub_component_type == SubComponentType.PREFILL:
            new_count = max(0, state.num_prefill_workers - 1)
            await self._update_scaling_decision(num_prefill=new_count)
        elif sub_component_type == SubComponentType.DECODE:
            new_count = max(0, state.num_decode_workers - 1)
            await self._update_scaling_decision(num_decode=new_count)
        elif sub_component_type == SubComponentType.ENCODER:
            current = self._desired_num_encode if self._desired_num_encode is not None else 0
            new_count = max(0, current - 1)
            await self._update_encode_decision(new_count)

        if blocking:
            await self._wait_for_scaling_completion()

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        num_prefill = None
        num_decode = None
        num_encode = None  # exp-4 R6

        for target_replica in target_replicas:
            if target_replica.sub_component_type == SubComponentType.PREFILL:
                num_prefill = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.DECODE:
                num_decode = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.ENCODER:
                # exp-4 R6: encoder targets are tracked Python-side
                # because the Rust binding's update_scaling_decision
                # signature is (num_prefill, num_decode) only.
                num_encode = target_replica.desired_replicas

        if num_prefill is None and num_decode is None and num_encode is None:
            return

        # Update prefill/decode through the Rust coordinator, encoder
        # through the Python-side tracker. Both calls are awaited in
        # parallel-friendly sequence (the Rust call is fire-and-forget
        # for the encoder side).
        if num_prefill is not None or num_decode is not None:
            await self._update_scaling_decision(
                num_prefill=num_prefill, num_decode=num_decode
            )
        if num_encode is not None:
            await self._update_encode_decision(num_encode)

        if blocking:
            await self._wait_for_scaling_completion()

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ):
        """Validate the deployment"""
        pass

    async def wait_for_deployment_ready(self, include_planner: bool = True):
        """Wait for the deployment to be ready"""
        await self._wait_for_scaling_completion()

    async def get_model_name(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> str:
        """Get the model name from the deployment"""
        del require_prefill, require_decode
        return self.model_name
