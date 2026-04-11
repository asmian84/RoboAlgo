# tests/test_elliott_wave/test_dataclasses.py
import pytest
from pattern_engine.elliott_wave import (
    WaveType, WaveDirection, Pivot, MonoWave, WaveCount
)


def make_pivot(idx: int, price: float, is_high: bool) -> Pivot:
    return Pivot(idx=idx, price=price, is_high=is_high)


def test_wavetype_has_all_variants():
    """All 12 pattern types must exist in WaveType enum."""
    assert WaveType.IMPULSE
    assert WaveType.IMPULSE_EXT_W1
    assert WaveType.IMPULSE_EXT_W3
    assert WaveType.IMPULSE_EXT_W5
    assert WaveType.TRUNCATED_5TH
    assert WaveType.ZIGZAG
    assert WaveType.FLAT_REGULAR
    assert WaveType.FLAT_EXPANDED
    assert WaveType.FLAT_RUNNING
    assert WaveType.CONTRACTING_TRIANGLE
    assert WaveType.EXPANDING_TRIANGLE
    assert WaveType.ENDING_DIAGONAL
    assert WaveType.LEADING_DIAGONAL


def test_monowave_internal_structure_defaults_to_indeterminate():
    p1 = make_pivot(0, 100.0, False)
    p2 = make_pivot(5, 110.0, True)
    mw = MonoWave(start=p1, end=p2)
    assert mw.internal_structure == "indeterminate"
    assert mw.classification_method == "unknown"
    assert mw.sub_pivot_count == 0


def test_monowave_direction_computed():
    p1 = make_pivot(0, 100.0, False)
    p2 = make_pivot(5, 90.0, True)
    mw = MonoWave(start=p1, end=p2)
    assert mw.direction == WaveDirection.DOWN
    assert mw.price_change == 10.0


def test_wavecount_has_counts_field():
    """WaveCount must have v2 fields for multi-count output."""
    p1 = make_pivot(0, 100.0, False)
    p2 = make_pivot(5, 110.0, True)
    mw = MonoWave(start=p1, end=p2)
    wc = WaveCount(
        waves=[mw],
        wave_type=WaveType.IMPULSE,
        direction=WaveDirection.UP,
    )
    assert hasattr(wc, "extension_wave")
    assert hasattr(wc, "invalidation_level")
    assert hasattr(wc, "alternation_quality")
    assert hasattr(wc, "position_bonus")
    assert wc.extension_wave is None
    assert wc.invalidation_level == 0.0
    assert wc.alternation_quality == "unknown"


# ── Task 2: MonoWave Builder ──────────────────────────────────────────────────

from pattern_engine.elliott_wave import build_monowaves


def test_build_monowaves_from_pivots():
    pivots = [
        Pivot(idx=0,  price=100.0, is_high=False),
        Pivot(idx=10, price=115.0, is_high=True),
        Pivot(idx=18, price=108.0, is_high=False),
        Pivot(idx=30, price=130.0, is_high=True),
        Pivot(idx=38, price=122.0, is_high=False),
        Pivot(idx=50, price=140.0, is_high=True),
    ]
    mws = build_monowaves(pivots)
    assert len(mws) == 5
    assert mws[0].direction == WaveDirection.UP
    assert mws[1].direction == WaveDirection.DOWN
    assert mws[0].price_change == pytest.approx(15.0)
    assert mws[0].bar_count == 10


def test_build_monowaves_empty_or_single():
    assert build_monowaves([]) == []
    assert build_monowaves([Pivot(0, 100.0, False)]) == []
