# tests/test_elliott_wave/test_candidates.py
import pytest
from pattern_engine.elliott_wave import (
    Pivot, MonoWave, WaveDirection, WaveType,
    build_monowaves, find_impulse_candidates
)


def _make_impulse_monowaves(ext_wave: int = 3) -> list[MonoWave]:
    """Build 5 MonoWaves forming a valid impulse. ext_wave=3 means W3 is biggest."""
    sizes = {1: 10, 2: 6.18, 3: 16.18, 4: 4.0, 5: 10}
    if ext_wave == 5:
        sizes = {1: 10, 2: 6.18, 3: 14, 4: 4.0, 5: 18}

    pivots = [Pivot(0, 100.0, False)]
    price = 100.0
    idx = 0
    directions = [1, -1, 1, -1, 1]
    for i, d in enumerate(directions, 1):
        idx += 10
        price += d * sizes[i]
        pivots.append(Pivot(idx, round(price, 4), d == 1))

    mws = build_monowaves(pivots)
    # Mark internal structures appropriately
    for j, mw in enumerate(mws):
        mw.internal_structure = "impulsive" if j % 2 == 0 else "corrective"
    return mws


def test_valid_impulse_w3_extended_found():
    mws = _make_impulse_monowaves(ext_wave=3)
    candidates = find_impulse_candidates(mws)
    assert len(candidates) >= 1
    assert candidates[0]["wave_type"] == WaveType.IMPULSE_EXT_W3


def test_valid_impulse_w5_extended_commodity():
    mws = _make_impulse_monowaves(ext_wave=5)
    candidates = find_impulse_candidates(mws, is_commodity=True)
    assert len(candidates) >= 1
    assert candidates[0]["wave_type"] in (WaveType.IMPULSE_EXT_W5, WaveType.IMPULSE)


def test_impulse_w4_overlaps_w1_rejected():
    """W4 end below W1 end — Rule 3 violation — must not appear in candidates."""
    pivots = [
        Pivot(0,  100.0, False),
        Pivot(10, 115.0, True),
        Pivot(18, 108.0, False),
        Pivot(30, 130.0, True),
        Pivot(38, 112.0, False),  # W4 end 112 < W1 end 115 → violation
        Pivot(50, 128.0, True),
    ]
    mws = build_monowaves(pivots)
    for j, mw in enumerate(mws):
        mw.internal_structure = "impulsive" if j % 2 == 0 else "corrective"
    candidates = find_impulse_candidates(mws)
    assert len(candidates) == 0


def test_impulse_w3_shortest_rejected():
    """W3 shorter than W1 and W5 — Rule 2 violation."""
    pivots = [
        Pivot(0,  100.0, False),
        Pivot(10, 118.0, True),   # W1 = 18
        Pivot(18, 110.0, False),  # W2
        Pivot(30, 120.0, True),   # W3 = 10 (shorter than W1=18 AND W5)
        Pivot(38, 114.0, False),  # W4
        Pivot(50, 135.0, True),   # W5 = 21
    ]
    mws = build_monowaves(pivots)
    for j, mw in enumerate(mws):
        mw.internal_structure = "impulsive" if j % 2 == 0 else "corrective"
    candidates = find_impulse_candidates(mws)
    assert len(candidates) == 0
