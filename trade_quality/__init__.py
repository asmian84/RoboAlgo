"""Trade Quality Scoring Engine — grades trade setups by confluence strength.

Grades:
    A+  (score 11–13) — highest quality,  100% base position
    A   (score  8–10) — high quality,      75% base position
    B   (score  5–7)  — moderate quality,  40% base position
    C   (score  0–4)  — low quality,        0% (do not trade)

Scoring factors (max 13 points):
    Liquidity sweep detected             +2
    Reversal Sniper confirmation         +3
    Time/cycle exhaustion                +2
    Volume spike                         +1
    Higher timeframe support/resistance  +2
    Wave/pattern correction phase        +1
    Favorable market regime              +2
"""

from trade_quality.quality_engine  import TradeQualityEngine
from trade_quality.grade_classifier import (
    classify_grade,
    compute_confidence,
    adjust_position_size,
    GradeSpec,
)
from trade_quality.scoring_factors import (
    FACTORS,
    MAX_SCORE,
    evaluate_factors,
)

__all__ = [
    "TradeQualityEngine",
    "classify_grade",
    "compute_confidence",
    "adjust_position_size",
    "GradeSpec",
    "FACTORS",
    "MAX_SCORE",
    "evaluate_factors",
]
