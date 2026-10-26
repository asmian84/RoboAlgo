"""
Apply the approved setups to today's scans.

The Finder's scans produce hits. This module tags each hit with which APPROVED setups it matches and
what the evidence for each says, so the scan carries what the learning found instead of ignoring it.
It tags; it does not re-rank or filter, and it adds nothing to a result when no approved setup applies.

WHAT "APPROVED" MEANS. Only setups in `registry/registry.json` with status "approved", which passed the
robustness gate in `learn.py`. Today those are trend setups, and the claim they carry is narrow: over
the next six months stocks matching them have earned MORE than the average liquid stock (a few percent
per half-year, before costs), while being LESS likely to double. They are not runner finders, and the
evidence says no runner setup has been found.

RELATIVE STRENGTH NEEDS A REFERENCE. A rule like "stronger than 70% of stocks" is a rank against the
whole market, which one scanned stock cannot compute alone. `registry/rs_reference.json` holds the
distribution of each momentum feature across liquid listed stocks, written by
`scripts/build_setup_dataset.py`. A rank is interpolated against it. The reference ages: it is stamped
with its date, and a result says so when it is more than `STALE_DAYS` old.

FAILURE IS VISIBLE. If the evidence cannot be computed (no history, no reference, any error) each
result carries `setup_evidence: None` and a `setup_evidence_error`, never a quiet empty list that reads
as "matches nothing".
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from setup_engine import features as ft
from setup_engine import learn as L

logger = logging.getLogger(__name__)

REFERENCE_PATH = L.REGISTRY_DIR / "rs_reference.json"
STALE_DAYS = 14
MIN_HISTORY_DAYS = 400            # ~80 weekly bars: enough for every 50-bar feature plus its 10-bar slope
_QUANTILES = np.round(np.arange(0, 100.5, 0.5), 1)


def build_reference(current: pd.DataFrame, path: Path = None) -> dict:
    """Distribution of each momentum source across today's liquid stocks, for interpolating ranks."""
    path = path or REFERENCE_PATH
    feats = {}
    for src in ft.RS_SOURCES:
        x = current[src].dropna().to_numpy(dtype=float)
        if len(x) < 200:
            continue
        feats[src] = {"values": [float(v) for v in np.quantile(x, _QUANTILES / 100.0)]}
    ref = {"asof": str(pd.Timestamp(current["date"].max()).date()), "n_stocks": int(len(current)),
           "levels": [float(q) for q in _QUANTILES], "features": feats}
    L.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref))
    return ref


def load_reference(path: Path = None) -> Optional[dict]:
    path = path or REFERENCE_PATH
    return json.loads(path.read_text()) if path.exists() else None


def rank_against(ref: dict, source: str, value: float) -> float:
    """Percentile rank (0-1) of `value` within the reference distribution of `source`."""
    f = ref["features"].get(source)
    if f is None or value is None or not np.isfinite(value):
        return float("nan")
    return float(np.interp(value, f["values"], ref["levels"]) / 100.0)


def approved_setups(reg: dict = None) -> list[dict]:
    reg = reg if reg is not None else L.load_registry()
    return [s for s in reg.get("setups", []) if s.get("status") == "approved" and s.get("rule")]


def _evidence_card(s: dict) -> dict:
    card = {"id": s["id"], "name": s.get("rule_text"), "claim": "earns more than the average liquid stock over 6 months, not more likely to double",
            "provisional": bool(s.get("provisional"))}
    ex = (s.get("excess_by_era") or {}).get("test")
    if ex:
        card.update(excess_6m_test=ex["excess_mean"], excess_6m_test_ci=[ex["excess_lo"], ex["excess_hi"]])
    liq = s.get("robustness") or {}
    if liq.get("liquid_excess_mean") is not None:
        card["excess_6m_liquid"] = liq["liquid_excess_mean"]
    return card


def match_setups(symbol: str, daily: pd.DataFrame, ref: dict, setups: list[dict]) -> Optional[list[dict]]:
    """Which approved setups does this stock match as of its last bar? None if it cannot be evaluated."""
    if daily is None or len(daily) < MIN_HISTORY_DAYS:
        return None
    cur = ft.current_features(symbol, daily)
    if cur is None:
        return None
    row = cur.to_dict()
    for src in ft.RS_SOURCES:
        row[f"rs_{src}"] = rank_against(ref, src, row.get(src))
    frame = pd.DataFrame([row])
    return [_evidence_card(s) for s in setups if L.Rule.from_json(s["rule"]).mask(frame)[0]]


def download_daily(symbols: Iterable[str], years: int = 2) -> dict[str, pd.DataFrame]:
    """Recent daily bars for a handful of stocks, in one batched call."""
    import yfinance as yf
    syms = sorted({s.replace(".", "-") for s in symbols})
    if not syms:
        return {}
    raw = yf.download(syms, period=f"{years}y", interval="1d", auto_adjust=False, group_by="ticker",
                      threads=True, progress=False)
    out = {}
    for s in syms:
        try:
            d = (raw[s] if len(syms) > 1 else raw)[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except Exception:
            continue
        if len(d):
            d = d.copy()
            d.columns = ["open", "high", "low", "close", "volume"]
            d.index = pd.DatetimeIndex(d.index).tz_localize(None)
            out[s] = d
    return out


def annotate(results: list[dict], daily_by_symbol: dict[str, pd.DataFrame] = None, reg: dict = None,
             ref: dict = None, fetch=download_daily) -> dict:
    """Add `setup_evidence` to every result in place. Returns a status dict describing what happened.

    `daily_by_symbol` may supply bars (the database path has them); anything missing is fetched in one
    batch. Never raises: a scan must not fail because its annotation did.
    """
    status = {"ok": True, "approved_setups": 0, "annotated": 0, "unevaluable": 0, "reference_asof": None, "error": None}
    try:
        setups = approved_setups(reg)
        status["approved_setups"] = len(setups)
        if not results:
            return status
        ref = ref if ref is not None else load_reference()
        if ref is None:
            raise RuntimeError("no relative-strength reference (run scripts/build_setup_dataset.py)")
        status["reference_asof"] = ref["asof"]
        age = (date.today() - pd.Timestamp(ref["asof"]).date()).days
        status["reference_stale"] = age > STALE_DAYS
        if not setups:
            for r in results:
                r["setup_evidence"] = []
            return status
        have = dict(daily_by_symbol or {})
        missing = [r["symbol"] for r in results if r["symbol"] not in have]
        if missing:
            have.update(fetch(missing))
        for r in results:
            cards = match_setups(r["symbol"], have.get(r["symbol"]) if r["symbol"] in have else have.get(r["symbol"].replace(".", "-")), ref, setups)
            if cards is None:
                r["setup_evidence"] = None
                r["setup_evidence_error"] = "not enough price history to evaluate"
                status["unevaluable"] += 1
            else:
                r["setup_evidence"] = cards
                status["annotated"] += 1
        return status
    except Exception as exc:                              # noqa: BLE001 - a scan must survive its annotation
        logger.warning("setup evidence unavailable: %s", exc)
        for r in results:
            r["setup_evidence"] = None
            r["setup_evidence_error"] = f"{type(exc).__name__}: {exc}"
        return {**status, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
