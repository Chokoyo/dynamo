# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight vLLM multimodal embedding-cache startup configuration."""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def configure_multimodal_embedding_cache(
    engine_args: object,
    *,
    route_to_encoder: bool,
    capacity_gb: float,
    namespace: str,
    component: str,
) -> None:
    """Configure vLLM's CPU embedding cache before engine creation.

    The connector is also enabled with separate encode workers so its
    ``ensure_cache_available`` hook can request only multimodal items not
    covered by authoritative KV state.
    """
    if capacity_gb <= 0:
        return

    from vllm.config import ECTransferConfig

    engine_id = f"{namespace}.{component}.backend.0"
    bridge_dir = os.environ.get("DYN_VLLM_EPD_BRIDGE_DIR")
    if not bridge_dir:
        bridge_dir = os.path.join(
            tempfile.gettempdir(),
            f"dynamo-vllm-epd-{os.getpid()}-{namespace}-{component}",
        )
        # vLLM may initialize the EC connector in a spawned EngineCore process.
        # Persist the parent-selected directory so the child connector and the
        # Dynamo request handler use the same filesystem bridge.
        os.environ["DYN_VLLM_EPD_BRIDGE_DIR"] = bridge_dir
    setattr(
        engine_args,
        "ec_transfer_config",
        ECTransferConfig(
            engine_id=engine_id,
            ec_role="ec_both",
            ec_connector="DynamoMultimodalEmbeddingCacheConnector",
            ec_connector_module_path=(
                "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
            ),
            ec_connector_extra_config={
                "multimodal_embedding_cache_capacity_gb": capacity_gb,
                "epd_bridge_dir": bridge_dir,
            },
        ),
    )
    logger.info(
        "Configured multimodal embedding cache: engine_id=%s, capacity=%.2f GB",
        engine_id,
        capacity_gb,
    )
