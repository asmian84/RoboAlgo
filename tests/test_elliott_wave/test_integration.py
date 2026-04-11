# tests/test_elliott_wave/test_integration.py
import numpy as np
import pandas as pd
import pytest
from pattern_engine.elliott_wave import analyze_multi_degree, clear_monowave_cache


def _make_clean_impulse_df(n_bars: int = 120) -> pd.DataFrame:
    """
    Generate a clean 5-wave impulse with W3 extension.
    Uses deterministic price path, not random, so tests are reproducible.
    """
    # W1 up 10%, W2 down 6.18%, W3 up 23%, W4 down 5%, W5 up 10%
    moves = [
        (20, +10.0),   # W1
        (12, -6.18),   # W2
        (30, +23.0),   # W3 (extended)
        (15, -5.0),    # W4
        (20, +10.0),   # W5
        (23, 0),       # flat tail
    ]
    prices = [100.0]
    for bars, move in moves:
        start = prices[-1]
        end   = start * (1 + move / 100)
        prices.extend(list(np.linspace(start, end, bars + 1))[1:])

    prices = prices[:n_bars]
    close = np.array(prices)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.003,
        "low":    close * 0.997,
        "close":  close,
        "volume": np.ones(len(close)) * 100_000,
    })


def test_analyze_returns_required_fields():
    df = _make_clean_impulse_df()
    clear_monowave_cache()
    result = analyze_multi_degree(df)
    assert "summary"     in result
    assert "confidence"  in result
    assert "phase"       in result
    assert "action"      in result
    assert "waves"       in result
    assert "projections" in result
    assert "counts"      in result
    assert "preferred"   in result


def test_analyze_counts_is_list():
    df = _make_clean_impulse_df()
    clear_monowave_cache()
    result = analyze_multi_degree(df)
    assert isinstance(result["counts"], list)


def test_analyze_no_nonsense_summary():
    """Summary must NOT be 'BULLISH IMPULSE REVERSE' or similar v1 garbage."""
    df = _make_clean_impulse_df()
    clear_monowave_cache()
    result = analyze_multi_degree(df)
    summary = result["summary"]
    assert "REVERSE" not in summary
    assert summary != "UNKNOWN"
    assert len(summary) > 3


def test_analyze_insufficient_data():
    df = pd.DataFrame({
        "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]
    })
    result = analyze_multi_degree(df)
    assert result["summary"] == "INSUFFICIENT DATA"
    assert result["counts"] == []


def test_analyze_confidence_between_0_and_100():
    df = _make_clean_impulse_df()
    clear_monowave_cache()
    result = analyze_multi_degree(df)
    assert 0.0 <= result["confidence"] <= 100.0


def test_preferred_count_is_highest_confidence():
    df = _make_clean_impulse_df()
    clear_monowave_cache()
    result = analyze_multi_degree(df)
    counts = result["counts"]
    if len(counts) >= 2:
        assert counts[0]["confidence"] >= counts[1]["confidence"]
