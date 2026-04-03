import pytest
import pandas as pd
import numpy as np
from smc_engine.order_block import detect_order_blocks, OrderBlock


def _make_df_with_bullish_ob():
    """Bearish candle followed by strong bullish impulse = bullish OB."""
    data = {
        "open":   [100, 101, 102, 101, 100, 99, 100, 105, 108, 112, 115, 118],
        "high":   [102, 103, 103, 102, 101, 100, 106, 109, 112, 115, 118, 120],
        "low":    [99,  100, 101, 100,  99,  98,  99, 104, 107, 111, 114, 117],
        "close":  [101, 102, 101, 100,  99,  100, 105, 108, 112, 115, 118, 120],
        "volume": [1000]*12,
    }
    return pd.DataFrame(data, index=pd.date_range("2026-01-01", periods=12, freq="h"))


class TestOrderBlockDetector:

    def test_detects_bullish_order_block(self):
        df = _make_df_with_bullish_ob()
        obs = detect_order_blocks(df)
        bullish = [ob for ob in obs if ob.direction == "bullish"]
        assert len(bullish) >= 1
        assert bullish[0].zone_top > bullish[0].zone_bottom
        assert bullish[0].mitigated is not None  # bool

    def test_returns_empty_on_flat_data(self):
        df = pd.DataFrame({
            "open": [100]*20, "high": [101]*20, "low": [99]*20,
            "close": [100.5]*20, "volume": [1000]*20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="h"))
        obs = detect_order_blocks(df)
        assert len(obs) == 0

    def test_ob_has_required_fields(self):
        df = _make_df_with_bullish_ob()
        obs = detect_order_blocks(df)
        if obs:
            ob = obs[0]
            assert hasattr(ob, "direction")
            assert hasattr(ob, "zone_top")
            assert hasattr(ob, "zone_bottom")
            assert hasattr(ob, "strength")
            assert hasattr(ob, "mitigated")
            assert hasattr(ob, "candle_index")
