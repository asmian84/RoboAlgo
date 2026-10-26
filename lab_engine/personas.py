"""
lab_engine/personas.py
────────────────────────────────────────────────────────────────────────────────
Rule-based investor personas — the Lab's answer to ai-hedge-fund's LLM analysts.

Each persona reads the same normalized fundamentals dict and independently
returns a verdict with the criteria that drove it. No LLM, no API cost,
fully deterministic — so a disagreement between personas is a real difference
in investing philosophy, not sampling noise.

Contract per persona:
  {
    "id", "name", "style",
    "verdict":    "BULLISH" | "BEARISH" | "NEUTRAL" | "INSUFFICIENT_DATA",
    "score":      -100..100   (signed conviction)
    "confidence": 0..100      (how much data backed the call)
    "criteria":   [{"label", "passed": True|False|None, "detail"}]
    "reasoning":  str
  }

Scoring: each persona defines weighted criteria. A criterion whose input is
missing is scored None and excluded from the denominator — so a persona
judged on 3 of 6 available criteria reports lower `confidence` rather than
silently treating missing data as a fail.
"""
from __future__ import annotations

from typing import Any, Callable

# ── Criterion helpers ─────────────────────────────────────────────────────


def _c(label: str, value: float | None, test: Callable[[float], bool], detail_fmt: str,
       weight: float = 1.0) -> dict[str, Any]:
    """Build one criterion result. `None` value → passed=None (excluded)."""
    if value is None:
        return {"label": label, "passed": None, "detail": "no data", "weight": weight}
    ok = test(value)
    return {"label": label, "passed": ok, "detail": detail_fmt.format(value), "weight": weight}


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _assemble(pid: str, name: str, style: str, criteria: list[dict],
              bull_at: float = 0.65, bear_at: float = 0.35) -> dict[str, Any]:
    """Turn scored criteria into a verdict + reasoning."""
    scored = [c for c in criteria if c["passed"] is not None]
    total_w = sum(c["weight"] for c in scored)

    if total_w == 0 or len(scored) < 2:
        return {
            "id": pid, "name": name, "style": style,
            "verdict": "INSUFFICIENT_DATA", "score": 0, "confidence": 0,
            "criteria": criteria,
            "reasoning": f"{name} needs at least 2 measurable criteria; "
                         f"only {len(scored)} available from this provider.",
        }

    passed_w = sum(c["weight"] for c in scored if c["passed"])
    ratio = passed_w / total_w

    if ratio >= bull_at:
        verdict = "BULLISH"
    elif ratio <= bear_at:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL"

    score = round((ratio - 0.5) * 200)
    # Confidence scales with how much of the persona's checklist we could run
    coverage = len(scored) / max(len(criteria), 1)
    confidence = round(coverage * 100)

    hits = [c["label"] for c in scored if c["passed"]]
    misses = [c["label"] for c in scored if not c["passed"]]
    parts = []
    if hits:
        parts.append("Passes: " + ", ".join(hits) + ".")
    if misses:
        parts.append("Fails: " + ", ".join(misses) + ".")
    parts.append(f"{len(scored)}/{len(criteria)} criteria measurable.")

    return {
        "id": pid, "name": name, "style": style,
        "verdict": verdict, "score": score, "confidence": confidence,
        "criteria": criteria,
        "reasoning": " ".join(parts),
    }


def _range_position(f: dict) -> float | None:
    """Where price sits in its 52-week range: 0.0 = at low, 1.0 = at high."""
    price, hi, lo = f.get("price"), f.get("week52_high"), f.get("week52_low")
    if price is None or hi is None or lo is None or hi <= lo:
        return None
    return max(0.0, min(1.0, (price - lo) / (hi - lo)))


# ── Personas ──────────────────────────────────────────────────────────────


def buffett(f: dict) -> dict[str, Any]:
    """Quality business at a fair price. Hates leverage, loves durable returns."""
    return _assemble("buffett", "Warren Buffett", "Quality at a fair price", [
        _c("ROE > 15%",           f.get("roe"),              lambda v: v > 0.15,  "ROE {0:.1%}", 1.5),
        _c("Net margin > 10%",    f.get("net_margin"),       lambda v: v > 0.10,  "margin {0:.1%}", 1.2),
        _c("Debt/Equity < 0.8",   f.get("debt_to_equity"),   lambda v: v < 0.8,   "D/E {0:.2f}", 1.3),
        _c("P/E < 25",            f.get("pe_ratio"),         lambda v: 0 < v < 25, "P/E {0:.1f}", 1.0),
        _c("Operating margin > 15%", f.get("operating_margin"), lambda v: v > 0.15, "op margin {0:.1%}", 1.0),
        _c("Large, established",  f.get("market_cap"),       lambda v: v > 10e9,  "cap ${0:,.0f}", 0.8),
    ])


def graham(f: dict) -> dict[str, Any]:
    """Strict quantitative value. Margin of safety or nothing."""
    pe, pb = f.get("pe_ratio"), f.get("pb_ratio")
    graham_number = (pe * pb) if (pe and pb and pe > 0 and pb > 0) else None
    return _assemble("graham", "Benjamin Graham", "Deep value, margin of safety", [
        _c("P/E < 15",            pe,                        lambda v: 0 < v < 15,  "P/E {0:.1f}", 1.5),
        _c("P/B < 1.5",           pb,                        lambda v: 0 < v < 1.5, "P/B {0:.2f}", 1.5),
        _c("Graham number < 22.5", graham_number,            lambda v: v < 22.5,    "P/E×P/B {0:.1f}", 1.3),
        _c("Current ratio > 2",   f.get("current_ratio"),    lambda v: v > 2.0,     "CR {0:.2f}", 1.2),
        _c("Pays a dividend",     f.get("dividend_yield"),   lambda v: v > 0,       "yield {0:.2%}", 0.8),
        _c("Debt/Equity < 1",     f.get("debt_to_equity"),   lambda v: v < 1.0,     "D/E {0:.2f}", 1.0),
    ])


def burry(f: dict) -> dict[str, Any]:
    """Contrarian deep value. Wants hated, cheap, and beaten-down."""
    rng = _range_position(f)
    return _assemble("burry", "Michael Burry", "Contrarian deep value", [
        _c("P/B < 1.2",           f.get("pb_ratio"),         lambda v: 0 < v < 1.2, "P/B {0:.2f}", 1.5),
        _c("P/E < 10",            f.get("pe_ratio"),         lambda v: 0 < v < 10,  "P/E {0:.1f}", 1.4),
        _c("Beaten down (bottom 35% of 52w)", rng,           lambda v: v < 0.35,    "range pos {0:.0%}", 1.3),
        _c("P/S < 1.5",           f.get("ps_ratio"),         lambda v: 0 < v < 1.5, "P/S {0:.2f}", 1.0),
        _c("Survivable leverage (D/E < 1.5)", f.get("debt_to_equity"), lambda v: v < 1.5, "D/E {0:.2f}", 1.2),
        _c("Squeeze fuel (short float > 5%)", f.get("short_percent_float"), lambda v: v > 0.05, "short {0:.1%}", 0.7),
    ])


def wood(f: dict) -> dict[str, Any]:
    """Innovation growth. Tolerates no earnings if the top line compounds."""
    rng = _range_position(f)
    sector = (f.get("sector") or "").lower()
    innov = sector if sector else None
    return _assemble("wood", "Cathie Wood", "Disruptive innovation growth", [
        _c("Revenue growth > 20%", f.get("revenue_growth"),  lambda v: v > 0.20,  "rev growth {0:.1%}", 1.8),
        _c("Earnings growth > 15%", f.get("earnings_growth"), lambda v: v > 0.15, "EPS growth {0:.1%}", 1.0),
        _c("Gross margin > 40%",  f.get("gross_margin"),     lambda v: v > 0.40,  "GM {0:.1%}", 1.3),
        _c("Momentum (top 40% of 52w)", rng,                 lambda v: v > 0.60,  "range pos {0:.0%}", 1.2),
        {"label": "Innovation sector", "weight": 1.4,
         **({"passed": any(k in innov for k in
                           ("technology", "communication", "healthcare", "consumer cyclical")),
             "detail": f.get("sector") or ""} if innov else {"passed": None, "detail": "no data"})},
        _c("High beta (>1.1)",    f.get("beta"),             lambda v: v > 1.1,   "beta {0:.2f}", 0.6),
    ])


def ackman(f: dict) -> dict[str, Any]:
    """Concentrated quality activist. Cash generation above all."""
    return _assemble("ackman", "Bill Ackman", "Concentrated quality / FCF", [
        _c("FCF margin > 12%",    f.get("fcf_margin"),       lambda v: v > 0.12,  "FCF margin {0:.1%}", 1.8),
        _c("Operating margin > 20%", f.get("operating_margin"), lambda v: v > 0.20, "op margin {0:.1%}", 1.4),
        _c("ROE > 12%",           f.get("roe"),              lambda v: v > 0.12,  "ROE {0:.1%}", 1.2),
        _c("Manageable leverage (D/E < 1.2)", f.get("debt_to_equity"), lambda v: v < 1.2, "D/E {0:.2f}", 1.1),
        _c("Mid/large cap (> $2B)", f.get("market_cap"),     lambda v: v > 2e9,   "cap ${0:,.0f}", 0.9),
        _c("Not richly priced (P/E < 30)", f.get("pe_ratio"), lambda v: 0 < v < 30, "P/E {0:.1f}", 0.8),
    ])


def druckenmiller(f: dict) -> dict[str, Any]:
    """Macro momentum. Rides strength; growth matters more than cheapness."""
    rng = _range_position(f)
    return _assemble("druckenmiller", "Stanley Druckenmiller", "Macro momentum", [
        _c("Strong momentum (top 30% of 52w)", rng,          lambda v: v > 0.70,  "range pos {0:.0%}", 1.8),
        _c("Revenue growth > 10%", f.get("revenue_growth"),  lambda v: v > 0.10,  "rev growth {0:.1%}", 1.3),
        _c("Earnings growth > 10%", f.get("earnings_growth"), lambda v: v > 0.10, "EPS growth {0:.1%}", 1.2),
        _c("Liquid (cap > $5B)",  f.get("market_cap"),       lambda v: v > 5e9,   "cap ${0:,.0f}", 1.0),
        _c("Profitable",          f.get("net_margin"),       lambda v: v > 0,     "net margin {0:.1%}", 0.9),
        _c("Beta > 0.9",          f.get("beta"),             lambda v: v > 0.9,   "beta {0:.2f}", 0.6),
    ])


PERSONAS: dict[str, Callable[[dict], dict]] = {
    "buffett":       buffett,
    "graham":        graham,
    "burry":         burry,
    "wood":          wood,
    "ackman":        ackman,
    "druckenmiller": druckenmiller,
}


def run_panel(fundamentals: dict, only: list[str] | None = None) -> dict[str, Any]:
    """
    Run every persona (or a subset) against one symbol's fundamentals.

    Returns the individual verdicts plus a panel consensus. The consensus is
    reported for comparison only — it is deliberately NOT fed into the live
    scanner score.
    """
    chosen = only or list(PERSONAS)
    verdicts = [PERSONAS[p](fundamentals) for p in chosen if p in PERSONAS]

    votable = [v for v in verdicts if v["verdict"] != "INSUFFICIENT_DATA"]
    bulls = sum(1 for v in votable if v["verdict"] == "BULLISH")
    bears = sum(1 for v in votable if v["verdict"] == "BEARISH")
    neutrals = sum(1 for v in votable if v["verdict"] == "NEUTRAL")

    if not votable:
        consensus, avg_score = "INSUFFICIENT_DATA", 0
    else:
        avg_score = round(sum(v["score"] for v in votable) / len(votable))
        if bulls > bears and avg_score > 10:
            consensus = "BULLISH"
        elif bears > bulls and avg_score < -10:
            consensus = "BEARISH"
        else:
            consensus = "NEUTRAL"

    # Disagreement is the interesting signal — a symbol every persona agrees
    # on is boring; a 3-3 split is where the styles actually differ.
    spread = max((v["score"] for v in votable), default=0) - \
             min((v["score"] for v in votable), default=0)

    return {
        "verdicts": verdicts,
        "consensus": consensus,
        "avg_score": avg_score,
        "tally": {"bullish": bulls, "bearish": bears, "neutral": neutrals,
                  "insufficient": len(verdicts) - len(votable)},
        "disagreement": spread,
    }


def list_personas() -> list[dict[str, str]]:
    """Persona roster for the UI — id, name, style."""
    return [
        {"id": pid, "name": fn({}).get("name", pid), "style": fn({}).get("style", "")}
        for pid, fn in PERSONAS.items()
    ]
