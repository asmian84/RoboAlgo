"""
tests/test_trade_signal_engine/test_universe.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the Universe Provider (trade_signal_engine.universe).

Network-dependent functions (get_full_universe, _get_all_exchange_tickers,
_enrich_float_and_si) are NOT tested here — see integration tests.
"""
from __future__ import annotations

import pytest

from trade_signal_engine.universe import (
    OPTIONS_FOCUS,
    SymbolCandidate,
    _BROAD_UNIVERSE,
    get_options_universe,
)


# ── _BROAD_UNIVERSE ────────────────────────────────────────────────────────────

class TestBroadUniverse:
    def test_is_non_empty_list(self):
        assert isinstance(_BROAD_UNIVERSE, list)
        assert len(_BROAD_UNIVERSE) > 0

    def test_has_more_than_100_symbols(self):
        assert len(_BROAD_UNIVERSE) > 100

    def test_all_entries_are_strings(self):
        assert all(isinstance(sym, str) for sym in _BROAD_UNIVERSE)

    def test_no_empty_strings(self):
        assert all(sym.strip() != "" for sym in _BROAD_UNIVERSE)


# ── SymbolCandidate dataclass ──────────────────────────────────────────────────

class TestSymbolCandidate:
    def test_create_with_required_fields(self):
        c = SymbolCandidate(symbol="AAPL", source="exchange")
        assert c.symbol == "AAPL"
        assert c.source == "exchange"

    def test_default_numeric_fields(self):
        c = SymbolCandidate(symbol="TSLA", source="tracked")
        assert c.price == 0.0
        assert c.change_pct == 0.0
        assert c.volume == 0.0
        assert c.avg_volume == 0.0
        assert c.volume_ratio == 1.0
        assert c.float_shares == 0.0
        assert c.short_pct == 0.0
        assert c.market_cap == 0.0

    def test_default_flags_is_empty_list(self):
        c = SymbolCandidate(symbol="SPY", source="curated")
        assert isinstance(c.flags, list)
        assert c.flags == []

    def test_flags_are_independent_per_instance(self):
        """Mutable default factory — each instance must get its own list."""
        c1 = SymbolCandidate(symbol="A", source="exchange")
        c2 = SymbolCandidate(symbol="B", source="exchange")
        c1.flags.append("volume_spike")
        assert "volume_spike" not in c2.flags

    def test_create_with_all_fields(self):
        c = SymbolCandidate(
            symbol="NVDA",
            source="exchange",
            price=850.0,
            change_pct=5.5,
            volume=50_000_000,
            avg_volume=25_000_000,
            volume_ratio=2.0,
            float_shares=2450.0,
            short_pct=1.2,
            market_cap=2_100_000.0,
            flags=["volume_spike", "big_mover"],
        )
        assert c.symbol == "NVDA"
        assert c.price == 850.0
        assert c.volume_ratio == 2.0
        assert "volume_spike" in c.flags

    def test_source_values(self):
        for src in ("exchange", "tracked", "curated"):
            c = SymbolCandidate(symbol="X", source=src)
            assert c.source == src


# ── get_options_universe ───────────────────────────────────────────────────────

class TestGetOptionsUniverse:
    def test_returns_non_empty_list(self):
        result = get_options_universe()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_same_object_as_options_focus(self):
        assert get_options_universe() is OPTIONS_FOCUS

    def test_all_entries_are_strings(self):
        assert all(isinstance(sym, str) for sym in get_options_universe())


# ── OPTIONS_FOCUS ──────────────────────────────────────────────────────────────

class TestOptionsFocus:
    def test_contains_spy(self):
        assert "SPY" in OPTIONS_FOCUS

    def test_contains_qqq(self):
        assert "QQQ" in OPTIONS_FOCUS

    def test_contains_aapl(self):
        assert "AAPL" in OPTIONS_FOCUS

    def test_is_list(self):
        assert isinstance(OPTIONS_FOCUS, list)

    def test_no_duplicates(self):
        assert len(OPTIONS_FOCUS) == len(set(OPTIONS_FOCUS))
