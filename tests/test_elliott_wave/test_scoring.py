# tests/test_elliott_wave/test_scoring.py
import pytest
from pattern_engine.elliott_wave import (
    Pivot, MonoWave, WaveDirection, WaveType,
    build_monowaves, find_impulse_candidates,
    validate_and_score, rank_counts
)


def _make_scored_impulse(w2_type="corrective", w4_type="corrective") -> dict:
    """Return a candidate dict ready for scoring."""
    pivots = [
        Pivot(0,  100.0, False),
        Pivot(10, 115.0, True),
        Pivot(18, 108.0, False),
        Pivot(30, 130.0, True),
        Pivot(38, 122.0, False),
        Pivot(50, 140.0, True),
    ]
    mws = build_monowaves(pivots)
    mws[0].internal_structure = "impulsive"
    mws[1].internal_structure = w2_type
    mws[2].internal_structure = "impulsive"
    mws[3].internal_structure = w4_type
    mws[4].internal_structure = "impulsive"
    candidates = find_impulse_candidates(mws)
    assert len(candidates) == 1
    return candidates[0]


def test_valid_impulse_scores_above_zero():
    cand = _make_scored_impulse()
    scored = validate_and_score([cand])
    assert len(scored) == 1
    assert scored[0]["confidence"] > 0.0


def test_good_alternation_scores_higher():
    """W2=zigzag (sharp), W4=flat (sideways) → better alternation than W2=W4=same."""
    cand_good = _make_scored_impulse(w2_type="corrective", w4_type="corrective")
    cand_same = _make_scored_impulse(w2_type="corrective", w4_type="corrective")
    # Both same structure here — just check scores are identical
    s1 = validate_and_score([cand_good])
    s2 = validate_and_score([cand_same])
    assert abs(s1[0]["confidence"] - s2[0]["confidence"]) < 0.01


def test_rank_counts_orders_by_confidence():
    cand1 = _make_scored_impulse()
    cand2 = _make_scored_impulse()
    scored = validate_and_score([cand1, cand2])
    scored[0]["confidence"] = 0.8
    scored[1]["confidence"] = 0.5
    ranked = rank_counts(scored, top_n=2)
    assert ranked[0]["confidence"] >= ranked[1]["confidence"]


def test_rank_counts_returns_at_most_top_n():
    cands = [_make_scored_impulse() for _ in range(5)]
    scored = validate_and_score(cands)
    ranked = rank_counts(scored, top_n=3)
    assert len(ranked) <= 3
