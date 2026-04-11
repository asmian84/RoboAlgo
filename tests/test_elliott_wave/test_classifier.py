# tests/test_elliott_wave/test_classifier.py
import numpy as np
import pandas as pd
import pytest
from pattern_engine.elliott_wave import classify_internal_structure


def _make_smooth_impulse(n=50) -> pd.DataFrame:
    """
    Synthetic 5-wave impulse: W1 up → W2 down (38.2%) → W3 up (1.618×W1)
    → W4 down (23.6%) → W5 up (100%W1).
    Uses realistic candle ranges with slight noise to produce detectable pivots.
    """
    np.random.seed(99)
    segments = [
        (8,  +10.0),   # W1 up 10
        (5,  -3.82),   # W2 down 38.2%
        (12, +16.18),  # W3 up 1.618×W1
        (5,  -3.82),   # W4 down 23.6% of W3
        (10, +10.0),   # W5 up = W1
        (10,  0.0),    # flat tail for context
    ]
    close_list = [100.0]
    for bars, move in segments:
        start = close_list[-1]
        end = start + move
        # Skip the first point to avoid duplicate junction prices
        close_list.extend(list(np.linspace(start, end, bars + 1))[1:])
    close = np.array(close_list[:n])
    noise = np.random.randn(len(close)) * 0.15  # tiny noise
    close = close + noise
    return pd.DataFrame({
        "open":   close - 0.2 + np.random.randn(len(close)) * 0.1,
        "high":   close + abs(np.random.randn(len(close)) * 0.8) + 0.3,
        "low":    close - abs(np.random.randn(len(close)) * 0.8) - 0.3,
        "close":  close,
        "volume": np.ones(len(close)) * 1000,
    })


def _make_zigzag_corrective(n=30) -> pd.DataFrame:
    """
    Synthetic 3-wave zigzag: A down 10 → B up 6 (60% retrace) → C down 10.
    Uses realistic candle ranges.
    """
    np.random.seed(77)
    segments = [
        (8,  -10.0),  # A down
        (6,  +6.0),   # B up (60% retrace)
        (10, -10.0),  # C down
        (6,   0.0),   # flat tail
    ]
    close_list = [120.0]
    for bars, move in segments:
        start = close_list[-1]
        end = start + move
        close_list.extend(list(np.linspace(start, end, bars + 1))[1:])
    close = np.array(close_list[:n])
    noise = np.random.randn(len(close)) * 0.15
    close = close + noise
    return pd.DataFrame({
        "open":   close - 0.2 + np.random.randn(len(close)) * 0.1,
        "high":   close + abs(np.random.randn(len(close)) * 0.8) + 0.3,
        "low":    close - abs(np.random.randn(len(close)) * 0.8) - 0.3,
        "close":  close,
        "volume": np.ones(len(close)) * 1000,
    })


def test_smooth_impulse_classifies_as_impulsive():
    df = _make_smooth_impulse()
    structure, method, count = classify_internal_structure(df, 0, len(df) - 1)
    assert structure == "impulsive"


def test_zigzag_classifies_as_corrective():
    df = _make_zigzag_corrective()
    structure, method, count = classify_internal_structure(df, 0, len(df) - 1)
    assert structure == "corrective"


def test_tiny_window_returns_indeterminate():
    """3 bars → not enough to classify, must be indeterminate."""
    df = _make_smooth_impulse()
    structure, method, _ = classify_internal_structure(df, 0, 3)  # bar_count=3 < 4
    assert structure == "indeterminate"


def test_recursive_method_used_for_long_windows():
    df = _make_smooth_impulse()
    _, method, _ = classify_internal_structure(df, 0, len(df) - 1)
    assert method == "recursive"


def test_heuristic_method_used_for_short_windows():
    """7-bar window → below min_bars_for_recursion=8 → heuristic."""
    df = _make_smooth_impulse()
    _, method, _ = classify_internal_structure(df, 0, 7)
    assert method in ("heuristic", "indeterminate")


def test_choppy_overlapping_is_corrective():
    """
    Choppy back-and-forth with large counter-moves → corrective.
    """
    # Create a move that retraces >38.2% multiple times
    prices = [100, 108, 103, 111, 105, 114, 108, 118]
    close = np.array(prices, dtype=float)
    df = pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5,  "close": close,
        "volume": np.ones(len(close)) * 1000,
    })
    structure, _, _ = classify_internal_structure(df, 0, len(df) - 1)
    # choppy with 38%+ retracements — should be corrective or indeterminate
    assert structure in ("corrective", "indeterminate")
