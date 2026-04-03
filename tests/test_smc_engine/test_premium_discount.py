"""
tests/test_smc_engine/test_premium_discount.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the Premium/Discount Zone Classifier + OTE zone detector
(smc_engine.premium_discount).
"""
from __future__ import annotations

import pandas as pd
import pytest

from smc_engine.premium_discount import ZoneAnalysis, analyze_zone


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_df(swing_low: float, swing_high: float, current_price: float, n: int = 60) -> pd.DataFrame:
    """
    Build a DataFrame whose high/low extremes define the desired dealing range
    and whose final close sits at *current_price*.

    The first candle anchors the swing low, the second anchors the swing high,
    and all remaining candles sit quietly at *current_price* so that
    analyze_zone sees exactly the range we want.
    """
    rows = []

    # candle 0: anchors swing low
    rows.append({
        "open": swing_low, "high": swing_low + 0.01,
        "low": swing_low, "close": swing_low,
    })
    # candle 1: anchors swing high
    rows.append({
        "open": swing_high, "high": swing_high,
        "low": swing_high - 0.01, "close": swing_high,
    })
    # remaining candles: price at current_price
    for _ in range(n - 2):
        rows.append({
            "open": current_price, "high": current_price,
            "low": current_price, "close": current_price,
        })

    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    df["volume"] = 1_000_000
    return df[["open", "high", "low", "close", "volume"]]


# ── Test 1: Price in DISCOUNT zone ───────────────────────────────────────────

def test_discount_zone():
    """
    Price well below the 50% equilibrium should be classified as DISCOUNT.

    Range: 100–200, equilibrium = 150.
    Price = 110 → zone_pct = 10% → DISCOUNT (below 45% threshold).
    """
    df = _make_df(swing_low=100.0, swing_high=200.0, current_price=110.0)
    result = analyze_zone(df, swing_lookback=60)

    assert result.zone == "DISCOUNT", f"Expected DISCOUNT, got {result.zone}"
    assert result.zone_pct == pytest.approx(10.0, abs=0.2)
    assert result.current_price == pytest.approx(110.0, abs=0.01)
    assert result.swing_low == pytest.approx(100.0, abs=0.01)
    assert result.swing_high == pytest.approx(200.0, abs=0.01)
    assert result.equilibrium == pytest.approx(150.0, abs=0.01)


# ── Test 2: Price in PREMIUM zone ────────────────────────────────────────────

def test_premium_zone():
    """
    Price well above the 50% equilibrium should be classified as PREMIUM.

    Range: 100–200, equilibrium = 150.
    Price = 185 → zone_pct = 85% → PREMIUM (above 55% threshold).
    """
    df = _make_df(swing_low=100.0, swing_high=200.0, current_price=185.0)
    result = analyze_zone(df, swing_lookback=60)

    assert result.zone == "PREMIUM", f"Expected PREMIUM, got {result.zone}"
    assert result.zone_pct == pytest.approx(85.0, abs=0.2)


# ── Test 3: Price at equilibrium ─────────────────────────────────────────────

def test_equilibrium_zone():
    """
    Price right at the 50% mid-point should be EQUILIBRIUM.

    Range: 100–200, equilibrium = 150.
    Price = 150 → zone_pct = 50% → EQUILIBRIUM (between 45–55%).
    """
    df = _make_df(swing_low=100.0, swing_high=200.0, current_price=150.0)
    result = analyze_zone(df, swing_lookback=60)

    assert result.zone == "EQUILIBRIUM", f"Expected EQUILIBRIUM, got {result.zone}"
    assert result.zone_pct == pytest.approx(50.0, abs=0.2)


# ── Test 4: OTE zone is correctly identified ─────────────────────────────────

def test_ote_zone_inside():
    """
    Price inside the 61.8–78.6% retracement (from swing high) should set
    in_ote = True.

    Range: 0–100.
      ote_top    = 100 - 100 * 0.618 = 38.2
      ote_bottom = 100 - 100 * 0.786 = 21.4

    Price = 30 lies inside [21.4, 38.2] → in_ote = True.
    """
    df = _make_df(swing_low=0.0, swing_high=100.0, current_price=30.0)
    result = analyze_zone(df, swing_lookback=60)

    assert result.ote_top    == pytest.approx(38.2, abs=0.1)
    assert result.ote_bottom == pytest.approx(21.4, abs=0.1)
    assert result.in_ote is True


def test_ote_zone_outside():
    """
    Price outside the OTE band should set in_ote = False.

    Range: 0–100, OTE band = [21.4, 38.2].
    Price = 75 (well above the band) → in_ote = False.
    """
    df = _make_df(swing_low=0.0, swing_high=100.0, current_price=75.0)
    result = analyze_zone(df, swing_lookback=60)

    assert result.in_ote is False


# ── Test 5: ZoneAnalysis has all required fields ─────────────────────────────

def test_zone_analysis_has_required_fields():
    """
    The returned ZoneAnalysis must expose every documented field with the
    correct type and sensible values.
    """
    df = _make_df(swing_low=100.0, swing_high=200.0, current_price=130.0)
    result = analyze_zone(df, swing_lookback=60)

    assert isinstance(result, ZoneAnalysis)

    # Type checks
    assert isinstance(result.swing_high,      float), "swing_high must be float"
    assert isinstance(result.swing_low,       float), "swing_low must be float"
    assert isinstance(result.equilibrium,     float), "equilibrium must be float"
    assert isinstance(result.current_price,   float), "current_price must be float"
    assert isinstance(result.zone,            str),   "zone must be str"
    assert isinstance(result.zone_pct,        float), "zone_pct must be float"
    assert isinstance(result.ote_top,         float), "ote_top must be float"
    assert isinstance(result.ote_bottom,      float), "ote_bottom must be float"
    assert isinstance(result.in_ote,          bool),  "in_ote must be bool"
    assert isinstance(result.discount_levels, list),  "discount_levels must be list"
    assert isinstance(result.premium_levels,  list),  "premium_levels must be list"

    # Semantic sanity
    assert result.zone in {"PREMIUM", "DISCOUNT", "EQUILIBRIUM"}
    assert 0.0 <= result.zone_pct <= 100.0
    assert result.swing_low < result.swing_high
    assert result.swing_low <= result.equilibrium <= result.swing_high
    assert result.ote_bottom <= result.ote_top
    # Fib levels: all discount levels below equilibrium, all premium above
    for lvl in result.discount_levels:
        assert lvl < result.equilibrium, f"Discount level {lvl} should be < equilibrium {result.equilibrium}"
    for lvl in result.premium_levels:
        assert lvl > result.equilibrium, f"Premium level {lvl} should be > equilibrium {result.equilibrium}"


# ── Test 6: Flat data (zero range) returns EQUILIBRIUM ───────────────────────

def test_flat_data_returns_equilibrium():
    """
    When all candles have the same OHLC, the dealing range is zero and the
    function must short-circuit to an EQUILIBRIUM result without crashing.
    """
    price = 100.0
    rows = [{"open": price, "high": price, "low": price, "close": price} for _ in range(30)]
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    df["volume"] = 1_000_000
    df = df[["open", "high", "low", "close", "volume"]]

    result = analyze_zone(df, swing_lookback=50)

    assert result.zone == "EQUILIBRIUM", f"Expected EQUILIBRIUM on flat data, got {result.zone}"
    assert result.zone_pct == pytest.approx(50.0)
    assert result.in_ote is False
    assert result.discount_levels == []
    assert result.premium_levels == []


# ── Test 7: Boundary zone thresholds ─────────────────────────────────────────

def test_zone_boundary_just_above_premium_threshold():
    """
    zone_pct just above 55 should be PREMIUM.

    Range: 0–100, price = 56 → zone_pct = 56% → PREMIUM.
    """
    df = _make_df(swing_low=0.0, swing_high=100.0, current_price=56.0)
    result = analyze_zone(df, swing_lookback=60)
    assert result.zone == "PREMIUM"


def test_zone_boundary_just_below_discount_threshold():
    """
    zone_pct just below 45 should be DISCOUNT.

    Range: 0–100, price = 44 → zone_pct = 44% → DISCOUNT.
    """
    df = _make_df(swing_low=0.0, swing_high=100.0, current_price=44.0)
    result = analyze_zone(df, swing_lookback=60)
    assert result.zone == "DISCOUNT"
