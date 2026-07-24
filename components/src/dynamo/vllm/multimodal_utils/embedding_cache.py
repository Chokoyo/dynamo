# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections import OrderedDict
from typing import NamedTuple


class CacheMutation(NamedTuple):
    stored: bool
    added_keys: list[str]
    removed_keys: list[str]


class EmbeddingCache:
    def __init__(self, capacity: int = 8):
        self.capacity = max(0, capacity)
        self.cache = OrderedDict()

    @classmethod
    def generate_hash_key(cls, *args):
        """
        Generate a hashable key based on the provided arguments.

        Args:
            *args: A variable number of arguments to generate the key.

        Returns:
            A string representing the hashable key.
        """
        key = hashlib.sha256()
        for arg in args:
            key.update(str(arg).encode("utf-8"))
        return key.hexdigest()

    def has_key(self, key):
        """
        Check if a key exists in the cache.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        return key in self.cache

    def set(self, key, value):
        """
        Store a key-value pair in the cache.

        Args:
            key: The key to store the value under.
            value: The value to store, expected to be a tuple.
        """
        return self.set_with_delta(key, value).stored

    def set_with_delta(self, key, value) -> CacheMutation:
        if self.capacity == 0:
            return CacheMutation(False, [], [])

        already_present = key in self.cache
        if already_present:
            self.cache.pop(key)

        removed_keys = []
        while len(self.cache) >= self.capacity:
            removed_key, _ = self.cache.popitem(last=False)
            removed_keys.append(removed_key)

        self.cache[key] = value
        return CacheMutation(
            True,
            [] if already_present else [key],
            removed_keys,
        )

    def get(self, key):
        """
        Retrieve the value associated with a key.

        Args:
            key: The key to look up.

        Returns:
            The value (tuple) associated with the key, or None if the key is not found.
        """
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
        return value

    def clear_with_delta(self) -> CacheMutation:
        removed_keys = list(self.cache)
        self.cache.clear()
        return CacheMutation(True, [], removed_keys)
