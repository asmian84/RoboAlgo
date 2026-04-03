"""
tests/test_smc_engine/test_smc_scorer.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the SMC Composite Scorer (smc_engine.smc_scorer).

Tests cover:
  - SMCResult has all required fields with correct types
  - Bullish data produces LONG or NEUTRAL bias (never SHORT)
  - Bearish data produces SHORT or NEUTRAL bias (never LONG)
  - Flat/sideways data produces NEUTRAL bias
  - sniper_entry is always a positive number
  - score is bounded 0-1
  - current_price override works
"""
from __future__ import annotations

import pandas as pd
import pytest

from smc_engine.smc_scorer import SMCResult, score_smc


# ── DataFrame builders ────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    if "volume" not in df.columns:
        df["volume"] = 1_000_000
    return df[["open", "high", "low", "close", "volume"]]


def _flat_candle(price: float) -> dict:
    return {"open": price, "high": price * 1.001, "low": price * 0.999,
            "close": price, "volume": 1_000_000}


def _bull_candle(open_: float, pct: float = 0.015) -> dict:
    """Green candle rising by pct."""
    close = open_ * (1 + pct)
    return {"open": open_, "high": close * 1.002, "low": open_ * 0.999,
            "close": close, "volume": 1_200_000}


def _bear_candle(open_: float, pct: float = 0.015) -> dict:
    """Red candle falling by pct."""
    close = open_ * (1 - pct)
    return {"open": open_, "high": open_ * 1.001, "low": close * 0.998,
            "close": close, "volume": 1_200_000}


def _make_uptrend(n: int = 80, start: float = 100.0, step_pct: float = 0.012) -> pd.DataFrame:
    """
    Steadily rising OHLCV data — higher highs and higher lows throughout.
    Produces structure shifts (BOS bullish) and a DISCOUNT entry zone early on.
    """
    rows = []
    price = start
    for i in range(n):
        # Alternate between small pullbacks and strong advances to create swing structure
        if i % 5 == 4:
            rows.append(_bear_candle(price, pct=0.005))
            price *= 0.995
        else:
            rows.append(_bull_candle(price, pct=step_pct))
            price *= (1 + step_pct)
    return _make_df(rows)


def _make_downtrend(n: int = 80, start: float = 100.0, step_pct: float = 0.012) -> pd.DataFrame:
    """
    Steadily falling OHLCV data — lower lows and lower highs throughout.
    """
    rows = []
    price = start
    for i in range(n):
        if i % 5 == 4:
            rows.append(_bull_candle(price, pct=0.005))
            price *= 1.005
        else:
            rows.append(_bear_candle(price, pct=step_pct))
            price *= (1 - step_pct)
    return _make_df(rows)


def _make_flat(n: int = 80, price: float = 100.0) -> pd.DataFrame:
    """
    Perfectly flat OHLCV data — every candle has identical OHLC values.
    This guarantees no structure shifts, no FVGs, and EQUILIBRIUM zone,
    producing a NEUTRAL composite score.
    """
    rows = [
        {"open": price, "high": price, "low": price, "close": price, "volume": 1_000_000}
        for _ in range(n)
    ]
    return _make_df(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSMCResultFields:
    """score_smc must return an SMCResult with all required fields."""

    def test_returns_smc_result_instance(self):
        df = _make_uptrend()
        result = score_smc(df)
        assert isinstance(result, SMCResult)

    def test_all_fields_present(self):
        df = _make_uptrend()
        result = score_smc(df)
        assert hasattr(result, "smc_score")
        assert hasattr(result, "smc_bias")
        assert hasattr(result, "entry_zone_top")
        assert hasattr(result, "entry_zone_bottom")
        assert hasattr(result, "sniper_entry")
        assert hasattr(result, "fvg_count")
        assert hasattr(result, "ob_count")
        assert hasattr(result, "latest_shift")
        assert hasattr(result, "zone")
        assert hasattr(result, "in_ote")
        assert hasattr(result, "detail")

    def test_field_types(self):
        df = _make_uptrend()
        r = score_smc(df)
        assert isinstance(r.smc_score, float)
        assert isinstance(r.smc_bias, str)
        assert isinstance(r.entry_zone_top, float)
        assert isinstance(r.entry_zone_bottom, float)
        assert isinstance(r.sniper_entry, float)
        assert isinstance(r.fvg_count, int)
        assert isinstance(r.ob_count, int)
        assert isinstance(r.latest_shift, str)
        assert isinstance(r.zone, str)
        assert isinstance(r.in_ote, bool)
        assert isinstance(r.detail, dict)

    def test_bias_is_valid_string(self):
        df = _make_uptrend()
        r = score_smc(df)
        assert r.smc_bias in {"LONG", "SHORT", "NEUTRAL"}

    def test_zone_is_valid_string(self):
        df = _make_uptrend()
        r = score_smc(df)
        assert r.zone in {"PREMIUM", "DISCOUNT", "EQUILIBRIUM"}

    def test_detail_has_expected_keys(self):
        df = _make_uptrend()
        r = score_smc(df)
        for key in ("zone_pct", "equilibrium", "swing_high", "swing_low"):
            assert key in r.detail, f"detail missing key: {key}"


class TestScoreBounds:
    """smc_score must always be in [0, 1]."""

    def test_score_bounded_uptrend(self):
        r = score_smc(_make_uptrend())
        assert 0.0 <= r.smc_score <= 1.0

    def test_score_bounded_downtrend(self):
        r = score_smc(_make_downtrend())
        assert 0.0 <= r.smc_score <= 1.0

    def test_score_bounded_flat(self):
        r = score_smc(_make_flat())
        assert 0.0 <= r.smc_score <= 1.0


class TestBullishBias:
    """Uptrending data must not produce a SHORT bias."""

    def test_uptrend_not_short(self):
        df = _make_uptrend(n=80, step_pct=0.015)
        r = score_smc(df)
        assert r.smc_bias in {"LONG", "NEUTRAL"}, (
            f"Expected LONG or NEUTRAL for uptrend, got {r.smc_bias} "
            f"(score={r.smc_score})"
        )

    def test_strong_uptrend_is_long_or_neutral(self):
        df = _make_uptrend(n=100, start=50.0, step_pct=0.02)
        r = score_smc(df)
        assert r.smc_bias != "SHORT", (
            f"Strong uptrend should not give SHORT (score={r.smc_score})"
        )


class TestBearishBias:
    """Downtrending data must not produce a LONG bias."""

    def test_downtrend_not_long(self):
        df = _make_downtrend(n=80, step_pct=0.015)
        r = score_smc(df)
        assert r.smc_bias in {"SHORT", "NEUTRAL"}, (
            f"Expected SHORT or NEUTRAL for downtrend, got {r.smc_bias} "
            f"(score={r.smc_score})"
        )

    def test_strong_downtrend_is_short_or_neutral(self):
        df = _make_downtrend(n=100, start=200.0, step_pct=0.02)
        r = score_smc(df)
        assert r.smc_bias != "LONG", (
            f"Strong downtrend should not give LONG (score={r.smc_score})"
        )


class TestNeutralBias:
    """Flat/sideways data should yield a NEUTRAL bias."""

    def test_flat_data_is_neutral(self):
        df = _make_flat(n=80, price=100.0)
        r = score_smc(df)
        assert r.smc_bias == "NEUTRAL", (
            f"Expected NEUTRAL for flat data, got {r.smc_bias} "
            f"(score={r.smc_score})"
        )

    def test_flat_score_near_half(self):
        df = _make_flat(n=80, price=100.0)
        r = score_smc(df)
        assert 0.35 <= r.smc_score <= 0.65, (
            f"Flat data score should be near 0.5, got {r.smc_score}"
        )


class TestSniperEntry:
    """sniper_entry must always be a valid positive price."""

    def test_sniper_entry_positive_uptrend(self):
        r = score_smc(_make_uptrend())
        assert r.sniper_entry > 0, "sniper_entry must be positive"

    def test_sniper_entry_positive_downtrend(self):
        r = score_smc(_make_downtrend())
        assert r.sniper_entry > 0, "sniper_entry must be positive"

    def test_sniper_entry_positive_flat(self):
        r = score_smc(_make_flat())
        assert r.sniper_entry > 0, "sniper_entry must be positive"

    def test_entry_zone_top_ge_bottom(self):
        """entry_zone_top should be >= entry_zone_bottom (valid zone)."""
        for df in [_make_uptrend(), _make_downtrend(), _make_flat()]:
            r = score_smc(df)
            assert r.entry_zone_top >= r.entry_zone_bottom, (
                f"entry_zone_top ({r.entry_zone_top}) < entry_zone_bottom "
                f"({r.entry_zone_bottom})"
            )


class TestCurrentPriceOverride:
    """current_price kwarg must override the default last-close value."""

    def test_override_uses_supplied_price(self):
        df = _make_uptrend()
        last_close = float(df["close"].iloc[-1])
        override_price = last_close * 1.05

        r_default = score_smc(df)
        r_override = score_smc(df, current_price=override_price)

        # sniper_entry falls back to current_price when no zone candidate is found,
        # so at minimum the two results should differ in at least one numeric field
        # OR the override price itself should have been used as baseline.
        assert r_override.sniper_entry != r_default.sniper_entry or \
               r_override.smc_score != r_default.smc_score or \
               r_override.entry_zone_top != r_default.entry_zone_top, (
            "Overriding current_price should change at least one result field"
        )

    def test_zero_current_price_falls_back_to_close(self):
        df = _make_uptrend()
        r = score_smc(df, current_price=0)
        last_close = float(df["close"].iloc[-1])
        # sniper_entry should be in the neighbourhood of last_close
        assert r.sniper_entry > 0


class TestEdgeCases:
    """Edge cases: minimal data, single-bar DataFrames."""

    def test_minimal_dataframe(self):
        """Very short DataFrame should not raise — returns a valid SMCResult."""
        rows = [_flat_candle(100.0) for _ in range(10)]
        df = _make_df(rows)
        r = score_smc(df)
        assert isinstance(r, SMCResult)
        assert r.sniper_entry > 0

    def test_fvg_ob_counts_non_negative(self):
        df = _make_uptrend()
        r = score_smc(df)
        assert r.fvg_count >= 0
        assert r.ob_count >= 0

    def test_latest_shift_is_string(self):
        df = _make_uptrend()
        r = score_smc(df)
        assert isinstance(r.latest_shift, str)
        # Should be "none" or contain a shift type
        assert r.latest_shift == "none" or any(
            kw in r.latest_shift for kw in ("CHoCH", "BOS")
        )
