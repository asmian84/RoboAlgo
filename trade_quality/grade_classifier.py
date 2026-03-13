"""Grade Classifier — maps raw quality scores to letter grades and size multipliers.

Grade thresholds
----------------
Score   Grade   Trade allowed   Position multiplier
──────  ──────  ──────────────  ───────────────────
11–13   A+      Yes             1.00  (100% base size)
 8–10   A       Yes             0.75  (75%  base size)
 5–7    B       Yes             0.40  (40%  base size)
 0–4    C       No              0.00  (skip trade)

Confidence is computed as ``score / max_score`` so that it is always
proportional to the evidence strength, not just the grade bucket.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Grade thresholds ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GradeSpec:
    label:              str
    min_score:          int
    trade_allowed:      bool
    position_multiplier:float


_GRADE_TABLE: list[GradeSpec] = [
    GradeSpec("A+", min_score=11, trade_allowed=True,  position_multiplier=1.00),
    GradeSpec("A",  min_score= 8, trade_allowed=True,  position_multiplier=0.75),
    GradeSpec("B",  min_score= 5, trade_allowed=True,  position_multiplier=0.40),
    GradeSpec("C",  min_score= 0, trade_allowed=False, position_multiplier=0.00),
]


def classify_grade(score: int, max_score: int = 13) -> GradeSpec:
    """Return the GradeSpec that corresponds to ``score``.

    Args:
        score:     Raw quality score (0 – max_score).
        max_score: Maximum possible score for normalisation.

    Returns:
        :class:`GradeSpec` for the matching grade bucket.
    """
    for spec in _GRADE_TABLE:
        if score >= spec.min_score:
            return spec
    return _GRADE_TABLE[-1]   # C — fallback


def compute_confidence(score: int, max_score: int = 13) -> float:
    """Normalise score to a 0–1 confidence value.

    The confidence is linearly proportional to score so that:
        - A score of 0  → confidence 0.0
        - A score of 13 → confidence 1.0

    This is independent of the grade bucket, giving smooth confidence
    gradients within each grade.
    """
    if max_score <= 0:
        return 0.0
    return round(min(max(score / max_score, 0.0), 1.0), 4)


def adjust_position_size(base_position: float, grade: GradeSpec) -> float:
    """Scale a base position size by the grade multiplier.

    Args:
        base_position: Unscaled position size (e.g. from position_sizer).
        grade:         :class:`GradeSpec` from :func:`classify_grade`.

    Returns:
        Adjusted position size.  Returns 0.0 for grade C.
    """
    return round(base_position * grade.position_multiplier, 2)


def grade_summary(spec: GradeSpec, score: int, max_score: int = 13) -> str:
    """One-line human-readable summary of the quality grade."""
    mult  = int(spec.position_multiplier * 100)
    trade = "TRADE" if spec.trade_allowed else "SKIP"
    return (
        f"{spec.label} ({score}/{max_score}) "
        f"→ {trade}, {mult}% size"
    )
