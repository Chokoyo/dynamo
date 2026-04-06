# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for KV router CLI argument configuration.

These tests verify that KvRouterArgGroup and KvRouterConfigBase correctly
configure router parameters, particularly around the router_queue_threshold
that controls when requests are queued vs dispatched immediately.

Issue #7836: Under single-turn workloads, decode workers can be starved when
active_tokens over-accounting keeps the all_workers_busy flag set.
Setting router_queue_threshold=0.0 enables aggressive queueing which is a
valid mitigation — the CLI and help text must not mislead operators into
thinking 0.0 is an invalid value.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# ---------------------------------------------------------------------------
# Module loader helpers — load kv_router_args without triggering the full
# dynamo.common package init chain (which needs compiled Rust bindings).
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parents[2] / "components" / "src"


def _load_module(rel_path: str, mod_name: str):
    """Load a Python file as a module by path, injecting it into sys.modules."""
    spec = importlib.util.spec_from_file_location(mod_name, _BASE / rel_path)
    assert spec is not None and spec.loader is not None, f"Cannot find {rel_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_stubs():
    """Create minimal stub packages so kv_router_args can be imported."""
    for name in ["dynamo", "dynamo.common", "dynamo.common.configuration"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _load_kv_router_args():
    _ensure_stubs()
    _load_module(
        "dynamo/common/configuration/config_base.py",
        "dynamo.common.configuration.config_base",
    )
    _load_module(
        "dynamo/common/configuration/utils.py",
        "dynamo.common.configuration.utils",
    )
    _load_module(
        "dynamo/common/configuration/arg_group.py",
        "dynamo.common.configuration.arg_group",
    )
    if "dynamo.common.configuration.groups" not in sys.modules:
        sys.modules["dynamo.common.configuration.groups"] = types.ModuleType(
            "dynamo.common.configuration.groups"
        )
    return _load_module(
        "dynamo/common/configuration/groups/kv_router_args.py",
        "dynamo.common.configuration.groups.kv_router_args",
    )


# Load once for the whole module
_kv_mod = _load_kv_router_args()
KvRouterArgGroup = _kv_mod.KvRouterArgGroup
KvRouterConfigBase = _kv_mod.KvRouterConfigBase
_KV_ROUTER_FIELDS = _kv_mod._KV_ROUTER_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser():
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    return parser


def _parse(args: list[str]) -> argparse.Namespace:
    return _make_parser().parse_args(args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouterQueueThreshold:
    """Tests verifying router_queue_threshold CLI behaviour.

    The Rust validation for this field is:
        #[validate(range(min = 0.0))]
    meaning 0.0 is explicitly allowed.  Setting threshold=0.0 tells the router
    to queue all requests the moment *any* prefill tokens are active, which is
    a useful configuration for decode workers in disaggregated mode.
    """

    def test_default_is_4_0(self):
        """Verify the default threshold is 4.0 (matching Rust KvRouterConfig::default)."""
        ns = _parse([])
        assert ns.router_queue_threshold == pytest.approx(4.0)

    def test_zero_is_accepted(self):
        """threshold=0.0 must be accepted — it enables maximum queueing sensitivity."""
        ns = _parse(["--router-queue-threshold", "0"])
        assert ns.router_queue_threshold == pytest.approx(0.0)

    def test_help_text_does_not_restrict_to_positive_only(self):
        """The help text must NOT say 'Must be > 0' since 0.0 is a valid configuration.

        The Rust struct uses #[validate(range(min = 0.0))] which allows 0.0.
        Operators setting threshold=0.0 to force aggressive queueing on
        decode workers should not be misled into thinking the value is invalid.

        This test demonstrates the bug described in issue #7836 where the
        misleading help text discouraged the use of threshold=0.0, a key
        mitigation for decode-worker starvation under single-turn workloads.
        """
        help_text = _make_parser().format_help()
        # The previous (incorrect) help said "Must be > 0." which forbids 0.
        # After the fix the help must not contain that restriction.
        assert "Must be > 0" not in help_text, (
            "router_queue_threshold help incorrectly says 'Must be > 0'. "
            "The value 0.0 is valid (and useful for aggressive queueing on "
            "decode workers). The help should say 'Must be >= 0.' instead."
        )

    def test_positive_value_is_accepted(self):
        """Standard threshold > 0 is accepted."""
        ns = _parse(["--router-queue-threshold", "2.5"])
        assert ns.router_queue_threshold == pytest.approx(2.5)

    def test_env_var_respected(self, monkeypatch):
        """DYN_ROUTER_QUEUE_THRESHOLD env var should override the default."""
        monkeypatch.setenv("DYN_ROUTER_QUEUE_THRESHOLD", "1.5")
        ns = _parse([])
        assert ns.router_queue_threshold == pytest.approx(1.5)

    def test_env_var_zero_is_respected(self, monkeypatch):
        """DYN_ROUTER_QUEUE_THRESHOLD=0 should result in threshold=0.0."""
        monkeypatch.setenv("DYN_ROUTER_QUEUE_THRESHOLD", "0")
        ns = _parse([])
        assert ns.router_queue_threshold == pytest.approx(0.0)


class TestKvRouterKwargs:
    """Tests for KvRouterConfigBase.kv_router_kwargs()."""

    def _make_config(self, **overrides):
        """Build a minimal KvRouterConfigBase instance via parse_args."""
        ns = _parse([])
        config = KvRouterConfigBase.__new__(KvRouterConfigBase)
        for k, v in vars(ns).items():
            setattr(config, k, v)
        for k, v in overrides.items():
            setattr(config, k, v)
        return config

    def test_kv_router_kwargs_includes_all_expected_fields(self):
        """kv_router_kwargs() must return all fields in _KV_ROUTER_FIELDS."""
        config = self._make_config()
        kwargs = config.kv_router_kwargs()
        for field in _KV_ROUTER_FIELDS:
            assert field in kwargs, f"Missing field in kv_router_kwargs(): {field!r}"

    def test_kv_router_kwargs_router_queue_threshold_zero(self):
        """kv_router_kwargs() must propagate router_queue_threshold=0.0 correctly."""
        config = self._make_config(router_queue_threshold=0.0)
        kwargs = config.kv_router_kwargs()
        assert kwargs["router_queue_threshold"] == pytest.approx(0.0)

    def test_kv_router_kwargs_router_track_prefill_tokens_false(self):
        """kv_router_kwargs() must propagate track_prefill_tokens=False.

        This is the key flag set by build_decode_router_override() in
        prefill_router/types.rs.  When False, decode workers should NOT
        accumulate active_tokens, preventing starvation under single-turn
        workloads (issue #7836).
        """
        config = self._make_config(router_track_prefill_tokens=False)
        kwargs = config.kv_router_kwargs()
        assert kwargs["router_track_prefill_tokens"] is False
