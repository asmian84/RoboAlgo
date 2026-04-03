"""
Unit tests for trade_signal_engine.collector.

Tests are offline only — no network calls are made.
Network-dependent behaviour is covered by integration tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_signal_engine.collector import (
    PROFILE_TFS,
    SymbolData,
    collect,
    collect_batch,
)


# ── PROFILE_TFS structure ─────────────────────────────────────────────────────

class TestProfileTfs:
    def test_has_all_three_profiles(self):
        assert set(PROFILE_TFS.keys()) == {"SCALP", "SWING", "POSITION"}

    def test_scalp_timeframes(self):
        tfs = [spec["tf"] for spec in PROFILE_TFS["SCALP"]]
        assert tfs == ["5m", "15m", "1h"]

    def test_swing_timeframes(self):
        tfs = [spec["tf"] for spec in PROFILE_TFS["SWING"]]
        assert tfs == ["1h", "4h", "1d"]

    def test_position_timeframes(self):
        tfs = [spec["tf"] for spec in PROFILE_TFS["POSITION"]]
        assert tfs == ["4h", "1d", "1wk"]

    def test_each_spec_has_tf_and_period(self):
        for profile, specs in PROFILE_TFS.items():
            for spec in specs:
                assert "tf" in spec, f"{profile} spec missing 'tf'"
                assert "period" in spec, f"{profile} spec missing 'period'"

    def test_scalp_periods(self):
        periods = {spec["tf"]: spec["period"] for spec in PROFILE_TFS["SCALP"]}
        assert periods["5m"] == "5d"
        assert periods["15m"] == "10d"
        assert periods["1h"] == "30d"

    def test_swing_periods(self):
        periods = {spec["tf"]: spec["period"] for spec in PROFILE_TFS["SWING"]}
        assert periods["1h"] == "60d"
        assert periods["4h"] == "60d"
        assert periods["1d"] == "120d"

    def test_position_periods(self):
        periods = {spec["tf"]: spec["period"] for spec in PROFILE_TFS["POSITION"]}
        assert periods["4h"] == "60d"
        assert periods["1d"] == "365d"
        assert periods["1wk"] == "365d"

    def test_each_profile_has_exactly_three_timeframes(self):
        for profile, specs in PROFILE_TFS.items():
            assert len(specs) == 3, f"{profile} should have 3 timeframe specs"


# ── SymbolData dataclass ──────────────────────────────────────────────────────

class TestSymbolData:
    def test_minimal_creation(self):
        sd = SymbolData(symbol="AAPL", profile="SWING")
        assert sd.symbol == "AAPL"
        assert sd.profile == "SWING"

    def test_defaults(self):
        sd = SymbolData(symbol="TSLA", profile="SCALP")
        assert sd.current_price == 0.0
        assert sd.bars == {}
        assert sd.options_chain is None
        assert sd.info == {}
        assert sd.error is None

    def test_bars_field_is_independent_per_instance(self):
        sd1 = SymbolData(symbol="AAPL", profile="SWING")
        sd2 = SymbolData(symbol="TSLA", profile="SWING")
        sd1.bars["1d"] = pd.DataFrame()
        assert "1d" not in sd2.bars, "bars dicts must not be shared between instances"

    def test_info_field_is_independent_per_instance(self):
        sd1 = SymbolData(symbol="AAPL", profile="SWING")
        sd2 = SymbolData(symbol="TSLA", profile="SWING")
        sd1.info["sector"] = "Tech"
        assert "sector" not in sd2.info

    def test_explicit_fields(self):
        dummy_df = pd.DataFrame({"close": [100.0]})
        sd = SymbolData(
            symbol="NVDA",
            profile="POSITION",
            current_price=500.0,
            bars={"1d": dummy_df},
            options_chain={"calls": [], "puts": []},
            info={"sector": "Technology"},
            error=None,
        )
        assert sd.current_price == 500.0
        assert "1d" in sd.bars
        assert sd.options_chain is not None
        assert sd.info["sector"] == "Technology"

    def test_error_field_can_be_set(self):
        sd = SymbolData(symbol="BAD", profile="SWING", error="Symbol not found")
        assert sd.error == "Symbol not found"


# ── collect() — offline behaviour ────────────────────────────────────────────

class TestCollectOffline:
    def test_unknown_profile_returns_error(self, monkeypatch):
        """collect() with an unrecognised profile must return immediately with an error."""
        # Patch yf.Ticker to make sure no network call is made
        import trade_signal_engine.collector as mod

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("AAPL", profile="UNKNOWN")
        assert sd.error is not None
        assert "Unknown profile" in sd.error
        assert sd.bars == {}

    def test_symbol_is_uppercased(self, monkeypatch):
        """collect() should normalise the symbol to uppercase."""
        import trade_signal_engine.collector as mod

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, **kw):
                return pd.DataFrame()
            @property
            def info(self):
                return {}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("aapl", profile="SWING")
        assert sd.symbol == "AAPL"

    def test_profile_is_uppercased(self, monkeypatch):
        """collect() should normalise the profile to uppercase."""
        import trade_signal_engine.collector as mod

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, **kw):
                return pd.DataFrame()
            @property
            def info(self):
                return {}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("AAPL", profile="swing")
        assert sd.profile == "SWING"

    def test_empty_history_sets_error(self, monkeypatch):
        """When all history calls return empty DataFrames, sd.error should be set."""
        import trade_signal_engine.collector as mod

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, **kw):
                return pd.DataFrame()
            @property
            def info(self):
                return {}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("FAKE", profile="SWING")
        assert sd.error is not None
        assert "No price data" in sd.error

    def test_bars_populated_from_history(self, monkeypatch):
        """When history returns data, bars dict should be populated."""
        import trade_signal_engine.collector as mod

        dummy = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1, freq="1h"),
        )

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, period, interval, **kw):
                # Return dummy data for non-4h intervals
                if interval != "1h":
                    return dummy.copy()
                return dummy.copy()
            @property
            def info(self):
                return {"marketCap": 1_000_000, "sector": "Technology"}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("AAPL", profile="SWING")
        # At least some bars should be populated (1h and 1d use direct history)
        assert len(sd.bars) > 0
        assert sd.error is None

    def test_current_price_set_from_bars(self, monkeypatch):
        """current_price should be taken from the first available tf's last close."""
        import trade_signal_engine.collector as mod

        close_val = 195.50
        dummy = pd.DataFrame(
            {
                "Open": [close_val],
                "High": [close_val + 1],
                "Low": [close_val - 1],
                "Close": [close_val],
                "Volume": [500],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="1h"),
        )

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, **kw):
                return dummy.copy()
            @property
            def info(self):
                return {}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("AAPL", profile="SWING")
        assert sd.current_price == pytest.approx(close_val)

    def test_fetch_options_false_leaves_chain_none(self, monkeypatch):
        """With fetch_options=False, options_chain must remain None."""
        import trade_signal_engine.collector as mod

        class _FakeTicker:
            def __init__(self, *a, **kw):
                pass
            def history(self, **kw):
                return pd.DataFrame()
            @property
            def info(self):
                return {}

        monkeypatch.setattr(mod.yf, "Ticker", _FakeTicker)

        sd = collect("AAPL", profile="SWING", fetch_options=False)
        assert sd.options_chain is None


# ── collect_batch() ───────────────────────────────────────────────────────────

class TestCollectBatch:
    def test_returns_list_of_symbol_data(self, monkeypatch):
        """collect_batch() should return one SymbolData per input symbol."""
        import trade_signal_engine.collector as mod

        def _fake_collect(symbol, profile, fetch_options):
            return SymbolData(symbol=symbol, profile=profile)

        monkeypatch.setattr(mod, "collect", _fake_collect)

        results = collect_batch(["AAPL", "TSLA", "NVDA"], profile="SWING")
        assert len(results) == 3
        symbols = {r.symbol for r in results}
        assert symbols == {"AAPL", "TSLA", "NVDA"}

    def test_options_fetched_only_for_specified_symbols(self, monkeypatch):
        """Only symbols in fetch_options_for should have fetch_options=True."""
        import trade_signal_engine.collector as mod

        fetched_options_for = []

        def _fake_collect(symbol, profile, fetch_options):
            if fetch_options:
                fetched_options_for.append(symbol)
            return SymbolData(symbol=symbol, profile=profile)

        monkeypatch.setattr(mod, "collect", _fake_collect)

        collect_batch(
            ["AAPL", "TSLA", "NVDA"],
            profile="SWING",
            fetch_options_for=["AAPL", "NVDA"],
        )
        assert set(fetched_options_for) == {"AAPL", "NVDA"}

    def test_empty_symbols_returns_empty_list(self, monkeypatch):
        import trade_signal_engine.collector as mod

        def _fake_collect(symbol, profile, fetch_options):
            return SymbolData(symbol=symbol, profile=profile)

        monkeypatch.setattr(mod, "collect", _fake_collect)

        results = collect_batch([], profile="SWING")
        assert results == []

    def test_collect_exception_captured_as_error(self, monkeypatch):
        """If collect() raises, collect_batch should wrap it in a SymbolData with error."""
        import trade_signal_engine.collector as mod

        def _boom(symbol, profile, fetch_options):
            raise RuntimeError("network timeout")

        monkeypatch.setattr(mod, "collect", _boom)

        results = collect_batch(["FAIL"], profile="SWING")
        assert len(results) == 1
        assert results[0].error == "network timeout"
        assert results[0].symbol == "FAIL"
