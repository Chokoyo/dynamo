# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native planner adapter subclasses (one per mode).

Each subclass sets ``require_prefill`` / ``require_decode`` and overrides
``_bootstrap_regression()`` and ``_apply_effects()``.  Everything else
(connector, Prometheus, FPM subscribers, tick loop) is in ``NativePlannerBase``.
"""

import logging

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import PlannerEffects
from dynamo.planner.monitoring.perf_metrics import fetch_pre_deployment_metrics

logger = logging.getLogger(__name__)


class PrefillPlanner(NativePlannerBase):
    """Prefill-only mode."""

    require_prefill = True
    require_decode = False

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.runtime_namespace,
                worker_info=self.prefill_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.PREFILL,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(prefill_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for prefill: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_prefill is None:
            return
        desired = effects.scale_to.num_prefill
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_prefill_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    component_name=self.prefill_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class DecodePlanner(NativePlannerBase):
    """Decode-only mode."""

    require_prefill = False
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.runtime_namespace,
                worker_info=self.decode_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.DECODE,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(decode_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for decode: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_decode is None:
            return
        desired = effects.scale_to.num_decode
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class EncoderPlanner(NativePlannerBase):
    """Encoder-pool scale-DOWN planner (exp-4).

    Sibling of ``PrefillPlanner`` / ``DecodePlanner``. Drives the
    ``_advance_load_encoder`` branch added in ``LoadScalingMixin``: the
    state machine accumulates ``encode_in_flight == 0`` ticks and emits a
    ``num_encode = N-1`` decision after K consecutive idle ticks.

    Defensive behavior:
      - require_prefill / require_decode = False: the planner only manages
        the encoder pool; if the deployment has no encode sub-component,
        the state machine's "no encoder workers" guard turns this into a
        no-op (PlannerEffects.scale_to stays None each tick).
      - _bootstrap_regression is a no-op: there is no FPM-based regression
        for the encoder (R4 used 5-s nvidia-smi as a proxy; the per-tick
        gauge is ``encode_in_flight`` from the encoder worker handler,
        which is consumed by the state machine, not the regression).
      - _apply_effects emits a single TargetReplica only when the state
        machine produced ``num_encode``; otherwise it returns silently so
        the connector layer never sees an empty target list.

    Scale-UP is intentionally absent. R2/R3 confirmed the encoder absorbs
    the burst at <5 % util on both Qwen2.5-VL-3B and LLaVA-1.5-7B on GB200,
    so the value proposition on this hardware is reclaim-only. See
    RESULTS.md §R4.5 for the opportunity sizing (≈100 % reclaim during
    text-only phases on the R4 text-heavy trace).
    """

    require_prefill = False
    require_decode = False

    async def _bootstrap_regression(self) -> None:
        # The encoder controller is gauge-driven, not regression-driven.
        # No-op: PrefillPlanner / DecodePlanner load FPM regressors; the
        # encoder branch only needs the live ``encode_in_flight`` gauge.
        logger.info("EncoderPlanner: no regression bootstrap required")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_encode is None:
            return
        desired = effects.scale_to.num_encode
        # Defensive: only attempt to route the target through the connector
        # if the deployment actually exposes an encode worker info entry.
        # Connectors that don't know about SubComponentType.ENCODER will
        # simply ignore the target (see VirtualConnector.set_component_replicas
        # which filters by enum membership).
        encode_worker_info = getattr(self, "encode_worker_info", None)
        component_name = (
            encode_worker_info.k8s_name if encode_worker_info is not None else None
        )
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.ENCODER,
                    component_name=component_name,
                    desired_replicas=desired,
                )
            ]
        )


class AggPlanner(NativePlannerBase):
    """Aggregated mode (single engine type handles both prefill and decode)."""

    require_prefill = False
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.runtime_namespace,
                worker_info=self.decode_worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.DECODE,
                aic_spec=self.config.aic_interpolation,
            )
            self.state_machine.load_benchmark_fpms(agg_fpms=fpms)
        except Exception as e:
            if self.config.enable_throughput_scaling:
                raise
            logger.warning(f"No pre-deployment data for agg: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None or effects.scale_to.num_decode is None:
            return
        desired = effects.scale_to.num_decode
        if self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(desired)
        await self._apply_scaling_targets(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=desired,
                )
            ]
        )


class DisaggPlanner(NativePlannerBase):
    """Disaggregated mode (separate prefill and decode engines)."""

    require_prefill = True
    require_decode = True

    async def _bootstrap_regression(self) -> None:
        for component, kwarg in [
            (SubComponentType.PREFILL, "prefill_fpms"),
            (SubComponentType.DECODE, "decode_fpms"),
        ]:
            worker_info = (
                self.prefill_worker_info
                if component == SubComponentType.PREFILL
                else self.decode_worker_info
            )
            try:
                fpms = await fetch_pre_deployment_metrics(
                    runtime=self.runtime,
                    namespace=self.runtime_namespace,
                    worker_info=worker_info,
                    profile_results_dir=self.config.profile_results_dir,
                    component_type=component,
                    aic_spec=self.config.aic_interpolation,
                )
                self.state_machine.load_benchmark_fpms(**{kwarg: fpms})
            except Exception as e:
                if self.config.enable_throughput_scaling:
                    raise
                logger.warning(f"No pre-deployment data for {component.value}: {e}")

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        if effects.scale_to is None:
            return
        decision = effects.scale_to

        if decision.num_prefill is not None and self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_prefill_replicas.set(
                decision.num_prefill
            )
        if decision.num_decode is not None and self.prometheus_port != 0:
            self.prometheus_metrics.predicted_num_decode_replicas.set(
                decision.num_decode
            )

        targets = []
        if decision.num_prefill is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    component_name=self.prefill_worker_info.k8s_name,
                    desired_replicas=decision.num_prefill,
                )
            )
        if decision.num_decode is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name=self.decode_worker_info.k8s_name,
                    desired_replicas=decision.num_decode,
                )
            )
        await self._apply_scaling_targets(targets)
