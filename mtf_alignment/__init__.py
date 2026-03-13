"""Multi-Timeframe Alignment Engine — quantifies signal agreement across timeframes.

The engine evaluates whether the regime bias from each timeframe in the
model hierarchy agrees with the top-level bias direction.

Trading models:
    DAY_TRADE   → bias=1h   (50%), setup=15m (30%), entry=5m  (20%)
    SWING_TRADE → bias=Daily(50%), setup=4h  (30%), entry=1h  (20%)
    INVESTMENT  → bias=Weekly(50%),setup=Daily(30%),entry=4h  (20%)

Alignment score interpretation:
    ≥ 0.80 — strong alignment  (all or most TFs agree)
    0.50–0.79 — moderate (bias + at least one lower TF agree)
    < 0.50  — weak / conflicting (do not open new positions)

Output format::

    {
        "alignment_score":        float,      # 0–1 weighted
        "bias_direction":         "LONG" | "SHORT" | "NEUTRAL",
        "model":                  str,
        "aligned_timeframes":     list[str],
        "conflicting_timeframes": list[str],
        "timeframe_details":      dict,       # per-TF regime breakdown
        "trade_allowed":          bool,
    }
"""

from mtf_alignment.mtf_alignment_engine import (
    MTFAlignmentEngine,
    compute_alignment,
    MODEL_DAY_TRADE,
    MODEL_SWING_TRADE,
    MODEL_INVESTMENT,
)

__all__ = [
    "MTFAlignmentEngine",
    "compute_alignment",
    "MODEL_DAY_TRADE",
    "MODEL_SWING_TRADE",
    "MODEL_INVESTMENT",
]
