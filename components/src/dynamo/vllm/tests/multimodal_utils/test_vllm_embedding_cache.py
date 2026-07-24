# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.vllm.multimodal_utils.embedding_cache import CacheMutation, EmbeddingCache


def test_embedding_cache_reports_lru_add_and_eviction_deltas():
    cache = EmbeddingCache(capacity=2)

    assert cache.set_with_delta("a", 1) == CacheMutation(True, ["a"], [])
    assert cache.set_with_delta("b", 2) == CacheMutation(True, ["b"], [])
    assert cache.get("a") == 1

    mutation = cache.set_with_delta("c", 3)

    assert mutation == CacheMutation(True, ["c"], ["b"])
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_embedding_cache_replacement_does_not_reannounce_residency():
    cache = EmbeddingCache(capacity=2)
    cache.set("a", 1)

    mutation = cache.set_with_delta("a", 2)

    assert mutation == CacheMutation(True, [], [])
    assert cache.get("a") == 2


def test_embedding_cache_clear_reports_all_resident_keys():
    cache = EmbeddingCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)

    mutation = cache.clear_with_delta()

    assert mutation == CacheMutation(True, [], ["a", "b"])
    assert cache.cache == {}
