# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Callable, Literal, Optional, Protocol

import sglang as sgl
import torch
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

from dynamo._core import Client, Context
from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import EMBEDDING_RECEIVER_FACTORIES, TransferRequest
from dynamo.common.multimodal_epd import MMSourceKind
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.llm import MultimodalEmbeddingCachePublisher
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_epd import (
    enforced_routing_plan,
    extract_media_objects,
    parse_planned_media_objects,
    target_prefill_worker,
    validate_object_response_indices,
)
from dynamo.sglang.protocol import (
    DisaggSglangMultimodalRequest,
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SglangEpdObjectRequest,
    SglangEpdObjectResponse,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"


class MultimodalConfig:
    """Configuration specific to multimodal processing"""

    EMBEDDINGS_DTYPE = torch.float16
    EMBEDDINGS_DEVICE = "cpu"


class EmbeddingsProcessorLike(Protocol):
    async def process_embeddings(
        self, request: SglangMultimodalRequest
    ) -> tuple[torch.Tensor, int]: ...

    def create_multimodal_image_item(
        self,
        embeddings: torch.Tensor,
        image_grid_thw: list[Any],
    ) -> dict[str, Any]: ...

    def create_multimodal_video_item(
        self,
        embeddings: torch.Tensor,
        video_grid_thw: list[Any],
        second_per_grid_ts: list[float] | None = None,
        video_timestamps: list[list[float]] | None = None,
    ) -> dict[str, Any]: ...


class SglangUtils:
    """General SGLang utilities (not multimodal-specific)"""

    @staticmethod
    def build_sampling_params(request: SglangMultimodalRequest) -> dict:
        """Build sampling parameters for SGLang engine (generic functionality)"""
        sampling_params = {}

        # Extract sampling options from request
        sampling_options = request.request.sampling_options
        stop_conditions = request.request.stop_conditions

        if sampling_options.temperature is not None:
            sampling_params["temperature"] = sampling_options.temperature
        if sampling_options.top_p is not None:
            sampling_params["top_p"] = sampling_options.top_p
        if sampling_options.top_k is not None:
            sampling_params["top_k"] = sampling_options.top_k
        if sampling_options.n is not None:
            sampling_params["n"] = sampling_options.n
        if stop_conditions.max_tokens:
            sampling_params["max_new_tokens"] = stop_conditions.max_tokens
        if stop_conditions.ignore_eos:
            sampling_params["ignore_eos"] = stop_conditions.ignore_eos

        logger.debug(f"Sampling params: {sampling_params}")
        return sampling_params


class EmbeddingsProcessor:
    """Handles multimodal embeddings processing and multimodal item creation"""

    def __init__(self, embedding_transfer_mode: EmbeddingTransferMode):
        receiver = EMBEDDING_RECEIVER_FACTORIES.get(embedding_transfer_mode)
        if receiver is None:
            raise ValueError(
                f"Invalid embedding transfer mode: {embedding_transfer_mode}"
            )
        self.embedding_receiver = receiver()

    async def process_embeddings(
        self, request: SglangMultimodalRequest
    ) -> tuple[torch.Tensor, int]:
        """Process one concatenated embedding tensor from serialized request."""
        logger.debug(f"Processing embeddings with shape: {request.embeddings_shape}")

        multimodal_groups = request.multimodal_inputs
        if not multimodal_groups:
            raise ValueError("multimodal_inputs is required")

        transfer_request = request.transfer_payload
        if transfer_request is None:
            raise ValueError("transfer_payload is required on request")

        if not isinstance(transfer_request, TransferRequest):
            transfer_request = TransferRequest.model_validate(transfer_request)

        embeddings_shape = request.embeddings_shape or tuple(
            transfer_request.embeddings_shape
        )
        if len(embeddings_shape) < 2:
            raise ValueError(f"Invalid embeddings shape: {embeddings_shape}")

        tensor_id, embeddings = await self.embedding_receiver.receive_embeddings(
            transfer_request
        )
        return embeddings, tensor_id

    def release_embeddings(self, tensor_id: int) -> None:
        self.embedding_receiver.release_tensor(tensor_id)

    @staticmethod
    def _create_processor_output_item(
        embeddings: torch.Tensor,
        grid_key: Literal["image_grid_thw", "video_grid_thw"],
        grid_values: list[Any],
        modality: Literal["IMAGE", "VIDEO"],
    ) -> dict[str, Any]:
        """Create shared processor_output fields for SGLang async_generate."""
        precomputed = embeddings.to(MultimodalConfig.EMBEDDINGS_DTYPE)
        grid_payload = torch.tensor(grid_values)

        mm_item: dict[str, Any] = {
            grid_key: grid_payload,
            "format": "processor_output",
            "precomputed_embeddings": precomputed,
            "modality": modality,
        }

        return mm_item

    @staticmethod
    def create_multimodal_image_item(
        embeddings: torch.Tensor,
        image_grid_thw: list[Any],
    ) -> dict[str, Any]:
        """Create an image processor_output mm_item for SGLang async_generate."""
        return EmbeddingsProcessor._create_processor_output_item(
            embeddings,
            "image_grid_thw",
            image_grid_thw,
            "IMAGE",
        )

    @staticmethod
    def create_multimodal_video_item(
        embeddings: torch.Tensor,
        video_grid_thw: list[Any],
        second_per_grid_ts: list[float] | None = None,
        video_timestamps: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Create a video processor_output mm_item for SGLang async_generate."""
        mm_item = EmbeddingsProcessor._create_processor_output_item(
            embeddings,
            "video_grid_thw",
            video_grid_thw,
            "VIDEO",
        )
        if second_per_grid_ts is not None:
            mm_item["second_per_grid_ts"] = torch.tensor(
                second_per_grid_ts, dtype=torch.float32
            )
        if video_timestamps is not None:
            # Keep per-video timestamp lists nested; Qwen VL indexes by video.
            mm_item["video_timestamps"] = video_timestamps
        return mm_item


class StreamProcessor:
    """Unified stream processing for SGLang responses"""

    @staticmethod
    async def process_sglang_stream(stream_source) -> AsyncIterator[str]:
        """Process SGLang stream output.

        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.
        """
        try:
            async for res in stream_source:
                try:
                    # With stream_output=True, output_ids contains only new tokens (disjoint)
                    output_ids = res.get("output_ids", [])
                    finish_reason = res.get("meta_info", {}).get("finish_reason")

                    # Empty, non-final chunks can happen during scheduler idle ticks.
                    # Keep waiting for the next chunk.
                    if not output_ids and not finish_reason:
                        continue

                    output = {
                        "token_ids": output_ids,
                        # Preserve SGLang's choice index for n>1 multimodal
                        # streams; older/non-n chunks are choice 0.
                        "index": res.get("index") or 0,
                        "text": res.get("text", ""),
                        "finished": False,
                    }

                    if finish_reason:
                        # For n > 1, choices can finish independently and SGLang
                        # may continue emitting chunks for other choice indices.
                        output.update(
                            {
                                "finish_reason": normalize_finish_reason(
                                    finish_reason.get("type", "stop")
                                ),
                                "finished": True,
                            }
                        )

                    yield json.dumps(output)

                except KeyError as e:
                    logger.error(
                        f"Missing key in SGLang response: {e}, available keys: {list(res.keys())}"
                    )
                    error_output = {
                        "token_ids": [],
                        "finish_reason": "error",
                        "error": f"Missing key: {e}",
                        "finished": True,
                    }
                    yield json.dumps(error_output)
                    break
                except Exception as e:
                    logger.error(f"Error processing SGLang response: {e}")
                    error_output = {
                        "token_ids": [],
                        "finish_reason": "error",
                        "error": str(e),
                        "finished": True,
                    }
                    yield json.dumps(error_output)
                    break

        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            error_output = {
                "token_ids": [],
                "finish_reason": "error",
                "error": str(e),
                "finished": True,
            }
            yield json.dumps(error_output)

    @staticmethod
    def create_bootstrap_info(
        bootstrap_host: str, bootstrap_port: int, bootstrap_room: int
    ) -> dict:
        """Create bootstrap info dictionary"""
        return {
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }


class ErrorResponseBuilder:
    """Standardized error response builder"""

    @staticmethod
    def build_error_response(error: Exception, extra_fields=None) -> str:
        """Build standardized error response"""
        response = {
            "token_ids": [],
            "finish_reason": "error",
            "error": str(error),
            "finished": True,
        }
        if extra_fields:
            response.update(extra_fields)
        return json.dumps(response)


async def _build_mm_items(
    request: SglangMultimodalRequest, embeddings_processor: EmbeddingsProcessorLike
) -> tuple[list[dict], list[dict], Optional[torch.Tensor], Optional[int]]:
    """Process embeddings and build multimodal items for SGLang.

    Returns:
        Tuple of (image_mm_items, video_data_items, combined_embeddings, tensor_id).
    """
    image_mm_items: list[dict] = []
    video_data_items: list[dict] = []

    encoded_groups: list[tuple[str, Any, int, float | None, list[float] | None]] = []

    for group in request.multimodal_inputs:
        if group.num_mm_tokens is not None and group.num_mm_tokens > 0:
            if group.image_grid_thw is not None:
                encoded_groups.append(
                    (
                        "IMAGE",
                        group.image_grid_thw,
                        group.num_mm_tokens,
                        None,
                        None,
                    )
                )
            elif group.video_grid_thw is not None:
                encoded_groups.append(
                    (
                        "VIDEO",
                        group.video_grid_thw,
                        group.num_mm_tokens,
                        group.second_per_grid_ts,
                        group.video_timestamps,
                    )
                )
            else:
                raise ValueError("Encoded multimodal group missing grid metadata")

    embeddings: Optional[torch.Tensor] = None
    tensor_id: Optional[int] = None

    if encoded_groups:
        embeddings, tensor_id = await embeddings_processor.process_embeddings(request)

        grouped_grids: dict[str, list[Any]] = {"IMAGE": [], "VIDEO": []}
        grouped_embeds: dict[str, list[torch.Tensor]] = {"IMAGE": [], "VIDEO": []}
        video_second_per_grid_ts: list[float] = []
        # SGLang expects one timestamp list per video in the grouped item.
        video_timestamps: list[list[float]] = []

        offset = 0
        for (
            modality,
            grid_item,
            token_count,
            second_per_grid_ts,
            timestamps,
        ) in encoded_groups:
            next_offset = offset + int(token_count)
            if next_offset > embeddings.shape[0]:
                raise ValueError("Encoded token counts exceed received embedding rows")
            grouped_grids[modality].append(grid_item)
            grouped_embeds[modality].append(embeddings[offset:next_offset])
            if modality == "VIDEO":
                if second_per_grid_ts is not None:
                    video_second_per_grid_ts.append(second_per_grid_ts)
                if timestamps is not None:
                    video_timestamps.append(timestamps)
            offset = next_offset

        if offset != embeddings.shape[0]:
            raise ValueError("Encoded token counts do not match received embeddings")

        if grouped_embeds["IMAGE"]:
            image_mm_items.append(
                embeddings_processor.create_multimodal_image_item(
                    torch.cat(grouped_embeds["IMAGE"], dim=0),
                    grouped_grids["IMAGE"],
                )
            )
        if grouped_embeds["VIDEO"]:
            video_group_count = len(grouped_grids["VIDEO"])
            if (
                video_second_per_grid_ts
                and len(video_second_per_grid_ts) != video_group_count
            ):
                raise ValueError(
                    "second_per_grid_ts must be present for every video group"
                )
            if video_timestamps and len(video_timestamps) != video_group_count:
                raise ValueError(
                    "video_timestamps must be present for every video group"
                )
            video_data_items.append(
                embeddings_processor.create_multimodal_video_item(
                    torch.cat(grouped_embeds["VIDEO"], dim=0),
                    grouped_grids["VIDEO"],
                    second_per_grid_ts=video_second_per_grid_ts or None,
                    video_timestamps=video_timestamps or None,
                )
            )

    return image_mm_items, video_data_items, embeddings, tensor_id


class MultimodalWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Multimodal worker handler for LLM inference with multimodal data.
    Handles both aggregated and disaggregated modes.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        prefill_client: Client | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, None, None, shutdown_event)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor(
            config.dynamo_args.embedding_transfer_mode
        )

        # Store serving mode and prefill client (like regular SGLang)
        self.serving_mode = config.serving_mode
        self.prefill_client = prefill_client

        # Validate prefill client for disaggregated mode
        if self.serving_mode == DisaggregationMode.DECODE:
            if self.prefill_client is None:
                raise ValueError(
                    "prefill_client must be provided when serving_mode is decode"
                )
            logger.info("Multimodal decode worker handler initialized")
        else:
            logger.info("Multimodal aggregated worker handler initialized")

    def _validate_and_parse_request(self, request) -> SglangMultimodalRequest:
        """Validate and parse incoming request"""
        if isinstance(request, str):
            request = json.loads(request)
        if (
            isinstance(request, dict)
            and "token_ids" in request
            and "request" not in request
        ):
            preprocessed = PreprocessedRequest.model_validate(request)
            groups = []
            for media_object in extract_media_objects(preprocessed):
                groups.append(
                    MultiModalGroup(
                        multimodal_input=MultiModalInput(
                            image_url=media_object.url
                            if media_object.modality == "IMAGE"
                            else None,
                            video_url=media_object.url
                            if media_object.modality == "VIDEO"
                            else None,
                        )
                    )
                )
            return SglangMultimodalRequest(
                request=preprocessed,
                multimodal_inputs=groups,
            )
        if type(request) is not SglangMultimodalRequest:
            request = SglangMultimodalRequest.model_validate(request)
        return request

    async def generate(
        self, request: SglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Generate response using SGLang with multimodal data
        Handles both aggregated and disaggregated modes (following regular SGLang DecodeWorkerHandler pattern)

        Args:
            request: Multimodal request with input and parameters.
            context: Context object for cancellation handling.
        """
        rng_pd = _nvtx.start_range("mm:pd:generate", color="green")
        rng_ttft = _nvtx.start_range("mm:pd:ttft", color="yellow")
        ttft_ended = False

        def _end_ttft() -> None:
            nonlocal ttft_ended
            if not ttft_ended:
                _nvtx.end_range(rng_ttft)
                ttft_ended = True

        try:
            request = self._validate_and_parse_request(request)

            # Route to appropriate generation method based on serving mode
            if self.serving_mode == DisaggregationMode.DECODE:
                rng_disagg = _nvtx.start_range("mm:pd:generate_disagg", color="red")
                try:
                    async for output in self._generate_disaggregated(
                        request, _end_ttft, context=context
                    ):
                        yield output
                finally:
                    _nvtx.end_range(rng_disagg)
            else:
                rng_agg = _nvtx.start_range("mm:pd:generate_agg", color="red")
                try:
                    async for output in self._generate_aggregated(
                        request, _end_ttft, context=context
                    ):
                        yield output
                finally:
                    _nvtx.end_range(rng_agg)

        except Exception as e:
            logger.error(f"Error in multimodal generation: {e}", exc_info=True)
            yield ErrorResponseBuilder.build_error_response(e)
        finally:
            _end_ttft()
            _nvtx.end_range(rng_pd)

    async def _generate_disaggregated(
        self,
        request: SglangMultimodalRequest,
        end_ttft: Callable[[], None],
        context=None,
    ) -> AsyncIterator[str]:
        """Handle disaggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")

        sampling_params = SglangUtils.build_sampling_params(request)

        # Request bootstrap info from prefill worker
        bootstrap_info = await self._get_bootstrap_from_prefill(
            request, sampling_params, context=context
        )

        trace_header = (
            context.trace_headers() if context and self.enable_trace else None
        )

        # Start decode generation with bootstrap info (no image data needed)
        decode_stream = await self.engine.async_generate(
            input_ids=bootstrap_info.get("input_ids", input_ids),
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=bootstrap_info["bootstrap_host"],
            bootstrap_port=bootstrap_info["bootstrap_port"],
            bootstrap_room=bootstrap_info["bootstrap_room"],
            external_trace_header=trace_header,
            rid=context.trace_id if context else None,
        )

        rng_first = _nvtx.start_range("mm:dec:first_token", color="purple")
        first_token = True
        try:
            async for output in StreamProcessor.process_sglang_stream(decode_stream):
                if first_token:
                    end_ttft()
                    _nvtx.end_range(rng_first)
                    first_token = False
                yield output
        finally:
            if first_token:
                end_ttft()
                _nvtx.end_range(rng_first)

    async def _generate_aggregated(
        self,
        request: SglangMultimodalRequest,
        end_ttft: Callable[[], None],
        context=None,
    ) -> AsyncIterator[str]:
        """Handle aggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")
        tensor_id: int | None = None
        try:
            sampling_params = SglangUtils.build_sampling_params(request)
            with _nvtx.annotate("mm:pd:load_multimodal", color="cyan"):
                (
                    image_mm_items,
                    video_data,
                    combined_embeddings,
                    tensor_id,
                ) = await _build_mm_items(request, self.embeddings_processor)

            if combined_embeddings is not None:
                logger.debug(
                    "Generated combined multimodal item with embeddings shape: "
                    f"{combined_embeddings.shape}"
                )
            else:
                logger.debug("No precomputed multimodal embeddings generated")
            logger.debug(f"Input token sequence length: {len(input_ids)}")

            trace_header = (
                context.trace_headers() if context and self.enable_trace else None
            )

            gen_params: dict[str, Any] = {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "stream": True,
                "external_trace_header": trace_header,
                "rid": context.trace_id if context else None,
            }
            if image_mm_items:
                gen_params["image_data"] = image_mm_items
            if video_data:
                gen_params["video_data"] = video_data

            agg_stream = await self.engine.async_generate(**gen_params)

            rng_first = _nvtx.start_range("mm:dec:first_token", color="purple")
            first_token = True
            try:
                async for output in StreamProcessor.process_sglang_stream(agg_stream):
                    if first_token:
                        if tensor_id is not None:
                            self.embeddings_processor.release_embeddings(tensor_id)
                            tensor_id = None
                        end_ttft()
                        _nvtx.end_range(rng_first)
                        first_token = False
                    yield output
            finally:
                if first_token:
                    end_ttft()
                    _nvtx.end_range(rng_first)

        except RuntimeError as e:
            if "shape mismatch" in str(e):
                logger.error(
                    "Shape mismatch error - this likely indicates a tokenization/embedding alignment issue"
                )
                logger.error(f"Request token IDs length: {len(input_ids)}")
                logger.error(f"Embeddings shape: {request.embeddings_shape}")
                logger.error(f"Token sequence preview: {input_ids[:20]}...")
                error_msg = (
                    f"Multimodal embedding alignment error: {str(e)}. "
                    f"This usually happens when the tokenization changes between requests. "
                    "Token count: "
                    f"{len(input_ids)}, Embedding shape: "
                    f"{request.embeddings_shape}"
                )
                yield ErrorResponseBuilder.build_error_response(RuntimeError(error_msg))
            else:
                yield ErrorResponseBuilder.build_error_response(e)
        finally:
            if tensor_id is not None:
                self.embeddings_processor.release_embeddings(tensor_id)

    async def _get_bootstrap_from_prefill(
        self, request: SglangMultimodalRequest, sampling_params: dict, context=None
    ) -> dict:
        """Get bootstrap info from prefill worker"""
        assert self.prefill_client is not None
        payload = DisaggSglangMultimodalRequest(
            request=request,
            sampling_params=sampling_params,
        ).model_dump_json()
        target_worker_id = target_prefill_worker(request.request)
        if target_worker_id is None:
            prefill_stream = await self.prefill_client.generate(
                payload,
                context=context,
            )
        else:
            logger.info(
                "SGLang EPD decode routing request to prefill worker %s",
                target_worker_id,
            )
            prefill_stream = await self.prefill_client.direct(
                payload,
                target_worker_id,
                context=context,
            )

        bootstrap_info = None
        async for info in prefill_stream:
            bootstrap_data = info.data() if hasattr(info, "data") else info
            if isinstance(bootstrap_data, str):
                bootstrap_info = json.loads(bootstrap_data)
            else:
                bootstrap_info = bootstrap_data
            break

        if not bootstrap_info:
            raise RuntimeError("No bootstrap info received from prefill worker")

        return bootstrap_info

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
        logger.info("Multimodal worker engine shutdown")


class MultimodalPrefillWorkerHandler(
    BaseWorkerHandler[DisaggSglangMultimodalRequest, str]
):
    """
    Multimodal prefill worker handler for disaggregated inference
    Processes multimodal inputs and coordinates with decode worker.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        encode_worker_client: Client,
        cache_publisher: MultimodalEmbeddingCachePublisher | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, None, None, shutdown_event)
        self.encode_worker_client = encode_worker_client
        self._cache_publisher = cache_publisher

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor(
            config.dynamo_args.embedding_transfer_mode
        )

        self._embedding_cache: MultimodalEmbeddingCacheManager | None = None
        capacity_gb = config.dynamo_args.multimodal_embedding_cache_capacity_gb
        if capacity_gb > 0:
            self._embedding_cache = MultimodalEmbeddingCacheManager(
                int(capacity_gb * 1024**3)
            )

        self.model = config.server_args.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=config.server_args.trust_remote_code
        )
        template = chat_templates[getattr(config.server_args, "chat_template")].copy()
        self.image_token_id = self._resolve_mm_token_id(
            tokenizer, template.image_token, "<|image_pad|>"
        )
        self.video_token_id = self._resolve_mm_token_id(
            tokenizer, getattr(template, "video_token", None), "<|video_pad|>"
        )

        # Get bootstrap info using BootstrapManager
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(engine)

        logger.info(
            f"Multimodal prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    @staticmethod
    def _resolve_mm_token_id(
        tokenizer, token: str | None, preferred: str
    ) -> int | None:
        candidates = [preferred]
        if token and token not in candidates:
            candidates.append(token)
        for candidate in candidates:
            token_id = tokenizer.convert_tokens_to_ids(candidate)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
        return None

    def _publish_cache_delta(
        self, added_keys: list[str], removed_keys: list[str]
    ) -> None:
        if self._cache_publisher is None or (not added_keys and not removed_keys):
            return
        try:
            self._cache_publisher.publish_delta(added_keys, removed_keys)
        except Exception:
            logger.warning(
                "Failed to publish SGLang prefill cache delta", exc_info=True
            )

    @staticmethod
    async def _first_object_response(stream) -> tuple[SglangEpdObjectResponse, Any]:
        response = await anext(stream)
        raw = response.data() if hasattr(response, "data") else response
        if isinstance(raw, str):
            parsed = SglangEpdObjectResponse.model_validate_json(raw)
        else:
            parsed = SglangEpdObjectResponse.model_validate(raw)
        return parsed, stream

    @staticmethod
    async def _drain_stream(stream) -> None:
        async for _ in stream:
            pass

    def _expand_object_placeholders(
        self, request: SglangMultimodalRequest, entries: list[CachedEmbedding]
    ) -> None:
        search_offsets = {"IMAGE": 0, "VIDEO": 0}
        media_objects = extract_media_objects(request.request)
        for media_object, entry in zip(media_objects, entries, strict=True):
            token_id = (
                self.image_token_id
                if media_object.modality == "IMAGE"
                else self.video_token_id
            )
            if token_id is None:
                raise ValueError(
                    f"{media_object.modality.lower()} token is not defined"
                )
            try:
                token_index = request.request.token_ids.index(
                    token_id, search_offsets[media_object.modality]
                )
            except ValueError as error:
                raise ValueError(
                    f"not enough {media_object.modality.lower()} placeholders"
                ) from error
            token_count = int(entry.tensor.shape[0])
            request.request.token_ids = (
                request.request.token_ids[:token_index]
                + [token_id] * token_count
                + request.request.token_ids[token_index + 1 :]
            )
            search_offsets[media_object.modality] = token_index + token_count

    async def _coordinate_epd_objects(
        self, request: SglangMultimodalRequest, context: Context | None
    ) -> tuple[list[dict], list[dict]]:
        available_workers = set(self.encode_worker_client.instance_ids())
        planned = parse_planned_media_objects(
            request.request,
            available_encode_worker_ids=available_workers,
        )
        if planned is None:
            raise ValueError("invalid or incomplete enforced SGLang EPD object plan")

        entries: list[CachedEmbedding | None] = [None] * len(planned)
        remote_by_worker: dict[int, list[Any]] = defaultdict(list)
        for item in planned:
            if item.plan.source_kind is MMSourceKind.P_LOCAL:
                cached = (
                    self._embedding_cache.get(item.cache_key)
                    if self._embedding_cache is not None
                    else None
                )
                if cached is not None:
                    logger.info(
                        "SGLang EPD prefill cache hit for object %s",
                        item.object_index,
                    )
                    entries[item.object_index] = cached
                    continue
                if item.url is None:
                    raise RuntimeError(
                        f"UUID-only object {item.object_index} disappeared from P-local cache"
                    )
                if not available_workers:
                    raise RuntimeError(
                        "stale P-local plan and no encode worker is live"
                    )
                fallback_worker = min(available_workers)
                remote_by_worker[fallback_worker].append(item)
            else:
                remote_by_worker[item.plan.source_worker_id].append(item)

        async def dispatch(worker_id: int, items: list[Any]):
            payload = SglangEpdObjectRequest(
                objects=[
                    {
                        "object_index": item.object_index,
                        "modality": item.modality,
                        "url": item.url,
                        "expected_cache_key": item.cache_key,
                    }
                    for item in items
                ]
            ).model_dump_json()
            attempted_workers: set[int] = set()
            candidate_worker = worker_id
            while True:
                attempted_workers.add(candidate_worker)
                stream = None
                try:
                    stream = await self.encode_worker_client.direct(
                        payload, candidate_worker, context=context
                    )
                    response, stream = await self._first_object_response(stream)
                    logger.info(
                        "SGLang EPD prefill received objects %s from encode worker %s",
                        [item.object_index for item in items],
                        candidate_worker,
                    )
                    return items, response, stream
                except Exception:
                    if stream is not None and hasattr(stream, "aclose"):
                        await stream.aclose()
                    fallback_workers = sorted(
                        set(self.encode_worker_client.instance_ids())
                        - attempted_workers
                    )
                    if not fallback_workers:
                        raise
                    if any(item.url is None for item in items):
                        raise RuntimeError(
                            "UUID-only embedding holder disappeared; source media is unavailable"
                        )
                    next_worker = fallback_workers[0]
                    logger.warning(
                        "SGLang EPD object dispatch to encode worker %s failed; "
                        "retrying URL objects on worker %s",
                        candidate_worker,
                        next_worker,
                        exc_info=True,
                    )
                    candidate_worker = next_worker

        dispatches = await asyncio.gather(
            *(
                dispatch(worker_id, items)
                for worker_id, items in remote_by_worker.items()
            )
        )
        expected_remote = [item for items, _, _ in dispatches for item in items]
        returned_parts = [
            part for _, response, _ in dispatches for part in response.parts
        ]
        validate_object_response_indices(
            expected_remote, [part.object_index for part in returned_parts]
        )
        expected_by_index = {item.object_index: item for item in expected_remote}

        async def receive_part(part):
            expected = expected_by_index[part.object_index]
            if part.modality != expected.modality:
                raise ValueError("encode response modality does not match request")
            if part.cache_key != expected.cache_key:
                raise ValueError("encode response cache key does not match request")
            (
                tensor_id,
                tensor,
            ) = await self.embeddings_processor.embedding_receiver.receive_embeddings(
                part.transfer_payload
            )
            try:
                if tuple(tensor.shape) != tuple(part.embeddings_shape):
                    raise ValueError(
                        "received embedding shape does not match descriptor"
                    )
                entry = CachedEmbedding(
                    tensor=tensor.detach().cpu().contiguous().clone(),
                    image_grid_thw=part.grid_thw if part.modality == "IMAGE" else None,
                    video_grid_thw=part.grid_thw if part.modality == "VIDEO" else None,
                    second_per_grid_ts=part.second_per_grid_ts,
                    video_timestamps=part.video_timestamps,
                )
            finally:
                self.embeddings_processor.release_embeddings(tensor_id)
            return part.object_index, part.cache_key, entry

        receive_tasks = [
            asyncio.create_task(receive_part(part)) for part in returned_parts
        ]
        try:
            loaded = await asyncio.gather(*receive_tasks)
            await asyncio.gather(
                *(self._drain_stream(stream) for _, _, stream in dispatches)
            )
        except BaseException:
            for task in receive_tasks:
                task.cancel()
            await asyncio.gather(*receive_tasks, return_exceptions=True)
            await asyncio.gather(
                *(
                    stream.aclose()
                    for _, _, stream in dispatches
                    if hasattr(stream, "aclose")
                ),
                return_exceptions=True,
            )
            raise
        for object_index, cache_key, entry in loaded:
            entries[object_index] = entry
            if self._embedding_cache is not None:
                mutation = self._embedding_cache.set_with_delta(cache_key, entry)
                self._publish_cache_delta(mutation.added_keys, mutation.removed_keys)

        if any(entry is None for entry in entries):
            raise ValueError("SGLang EPD object plan did not resolve every object")
        resolved = [entry for entry in entries if entry is not None]
        self._expand_object_placeholders(request, resolved)

        image_entries = [
            entry
            for item, entry in zip(planned, resolved, strict=True)
            if item.modality == "IMAGE"
        ]
        video_entries = [
            entry
            for item, entry in zip(planned, resolved, strict=True)
            if item.modality == "VIDEO"
        ]
        image_items: list[dict] = []
        video_items: list[dict] = []
        if image_entries:
            image_items.append(
                self.embeddings_processor.create_multimodal_image_item(
                    torch.cat([entry.tensor for entry in image_entries], dim=0),
                    [entry.image_grid_thw for entry in image_entries],
                )
            )
        if video_entries:
            video_items.append(
                self.embeddings_processor.create_multimodal_video_item(
                    torch.cat([entry.tensor for entry in video_entries], dim=0),
                    [entry.video_grid_thw for entry in video_entries],
                    [entry.second_per_grid_ts for entry in video_entries]
                    if all(
                        entry.second_per_grid_ts is not None for entry in video_entries
                    )
                    else None,
                    [entry.video_timestamps for entry in video_entries]
                    if all(
                        entry.video_timestamps is not None for entry in video_entries
                    )
                    else None,
                )
            )
        return image_items, video_items

    async def generate(
        self, disagg_request: DisaggSglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Handle prefill phase: process multimodal input and provide bootstrap info

        Args:
            disagg_request: Disaggregated multimodal request.
            context: Context object for cancellation handling.
        """
        rng_bootstrap = _nvtx.start_range("mm:prefill:bootstrap", color="yellow")
        bootstrap_ended = False

        def _end_bootstrap() -> None:
            nonlocal bootstrap_ended
            if not bootstrap_ended:
                _nvtx.end_range(rng_bootstrap)
                bootstrap_ended = True

        bootstrap_room = None
        try:
            # Validate and parse request
            disagg_request = self._validate_and_parse_disagg_request(disagg_request)

            prepared_mm_items: tuple[list[dict], list[dict]] | None = None
            if enforced_routing_plan(disagg_request.request.request) is not None:
                with _nvtx.annotate("mm:prefill:epd_coordinate", color="orange"):
                    prepared_mm_items = await self._coordinate_epd_objects(
                        disagg_request.request, context
                    )

            # Generate and return bootstrap info first (like regular SGLang)
            bootstrap_room = self._generate_bootstrap_room()
            bootstrap_info = {
                "bootstrap_host": self.bootstrap_host,
                "bootstrap_port": self.bootstrap_port,
                "bootstrap_room": bootstrap_room,
            }
            if prepared_mm_items is not None:
                bootstrap_info["input_ids"] = disagg_request.request.request.token_ids

            _end_bootstrap()
            yield json.dumps(bootstrap_info)

            # Process prefill generation
            await self._process_prefill_generation(
                disagg_request,
                bootstrap_room,
                context=context,
                prepared_mm_items=prepared_mm_items,
            )

        except Exception as e:
            logger.error(f"Error in prefill generation: {e}", exc_info=True)
            extra_fields = (
                {"bootstrap_room": bootstrap_room} if bootstrap_room is not None else {}
            )
            yield ErrorResponseBuilder.build_error_response(e, extra_fields)
        finally:
            _end_bootstrap()

    def _validate_and_parse_disagg_request(
        self, disagg_request
    ) -> DisaggSglangMultimodalRequest:
        """Validate and parse disaggregated request"""
        if type(disagg_request) is not DisaggSglangMultimodalRequest:
            if type(disagg_request) is str:
                disagg_request = DisaggSglangMultimodalRequest.model_validate_json(
                    disagg_request
                )
            else:
                disagg_request = DisaggSglangMultimodalRequest.model_validate(
                    disagg_request
                )
        return disagg_request

    async def _process_prefill_generation(
        self,
        disagg_request: DisaggSglangMultimodalRequest,
        bootstrap_room: int,
        context=None,
        prepared_mm_items: tuple[list[dict], list[dict]] | None = None,
    ):
        """Process multimodal input and start prefill generation"""
        # Get the SglangMultimodalRequest from the DisaggSglangMultimodalRequest
        request = disagg_request.request
        input_ids = request.request.token_ids
        sampling_params = disagg_request.sampling_params
        tensor_id: int | None = None

        # Process embeddings from encode worker using our embeddings processor
        if prepared_mm_items is None:
            with _nvtx.annotate("mm:prefill:load_multimodal", color="cyan"):
                (
                    image_mm_items,
                    video_data,
                    _,
                    tensor_id,
                ) = await _build_mm_items(request, self.embeddings_processor)
        else:
            image_mm_items, video_data = prepared_mm_items

        trace_header = (
            context.trace_headers() if context and self.enable_trace else None
        )

        # Start SGLang prefill generation (like regular SGLang)
        with _nvtx.annotate("mm:prefill:engine_async_generate", color="blue"):
            gen_params = {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "stream": True,
                "bootstrap_host": self.bootstrap_host,
                "bootstrap_port": self.bootstrap_port,
                "bootstrap_room": bootstrap_room,
                "external_trace_header": trace_header,
                "rid": context.trace_id if context else None,
            }

            if image_mm_items:
                gen_params["image_data"] = image_mm_items
            if video_data:
                gen_params["video_data"] = video_data

            results = await self.engine.async_generate(**gen_params)

        # Consume results without yielding (prefill doesn't return text, just coordinates)
        asyncio.create_task(self._consume_results(results, tensor_id))

    async def _consume_results(self, results, tensor_id: Optional[int]):
        """Consume prefill results without returning them (like regular SGLang)"""
        released = False
        try:
            async for _ in results:
                if tensor_id is not None and not released:
                    self.embeddings_processor.release_embeddings(tensor_id)
                    released = True
        finally:
            if tensor_id is not None and not released:
                self.embeddings_processor.release_embeddings(tensor_id)

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
        logger.info("Multimodal prefill engine shutdown")
