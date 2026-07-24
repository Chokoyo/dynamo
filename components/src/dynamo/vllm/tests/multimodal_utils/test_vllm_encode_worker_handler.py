# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from dynamo.common.multimodal.embedding_transfer import TransferRequest
from dynamo.vllm.multimodal_handlers.encode_worker_handler import (
    EmbeddingItem,
    _attach_transfer_metadata,
)
from dynamo.vllm.multimodal_utils.protocol import MultiModalGroup, MultiModalInput


def test_attach_transfer_metadata_preserves_requested_url():
    group = MultiModalGroup(
        multimodal_input=MultiModalInput(image_url="http://image/1")
    )
    embedding_item = EmbeddingItem(
        key="embedding-key",
        image_grid_thw=[[1, 2, 3]],
        embeddings=torch.zeros((1, 4, 8), dtype=torch.float16),
    )
    transfer_request = TransferRequest(
        embeddings_shape=[1, 4, 8],
        embedding_dtype_str="float16",
        serialized_request=7,
    )

    _attach_transfer_metadata(group, embedding_item, transfer_request)

    assert group.multimodal_input.image_url == "http://image/1"
    assert group.image_grid_thw == [[1, 2, 3]]
    assert group.embeddings_shape == (1, 4, 8)
    assert group.serialized_request == transfer_request
