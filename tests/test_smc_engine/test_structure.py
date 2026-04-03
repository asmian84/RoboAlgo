import pytest
import pandas as pd
import numpy as np
from smc_engine.structure import detect_structure_shifts, StructureShift


def _make_uptrend_then_choch():
    """Uptrend (HH/HL) followed by a break below last HL = bearish CHoCH."""
    prices = [100, 105, 103, 108, 106, 112, 109, 115, 112, 108, 105, 103]
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p + 2 for p in prices],
        "low": [p - 1 for p in prices],
        "close": [p + 0.5 for p in prices],
        "volume": [1000]*n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="h"))


def _make_uptrend_with_bos():
    """Steady uptrend with higher highs = bullish BOS."""
    prices = [100, 103, 101, 106, 103, 109, 106, 112, 109, 115]
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p + 2 for p in prices],
        "low": [p - 1 for p in prices],
        "close": [p + 1 for p in prices],
        "volume": [1000]*n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="h"))


class TestStructureShifts:

    def test_detects_bearish_choch(self):
        df = _make_uptrend_then_choch()
        shifts = detect_structure_shifts(df)
        chochs = [s for s in shifts if s.shift_type == "CHoCH"]
        assert len(chochs) >= 1
        assert any(s.direction == "bearish" for s in chochs)

    def test_detects_bullish_bos(self):
        df = _make_uptrend_with_bos()
        shifts = detect_structure_shifts(df)
        bos_list = [s for s in shifts if s.shift_type == "BOS"]
        assert len(bos_list) >= 1
        assert any(s.direction == "bullish" for s in bos_list)

    def test_has_required_fields(self):
        df = _make_uptrend_then_choch()
        shifts = detect_structure_shifts(df)
        if shifts:
            s = shifts[0]
            assert s.shift_type in ("CHoCH", "BOS")
            assert s.direction in ("bullish", "bearish")
            assert s.level > 0
            assert s.candle_index >= 0
