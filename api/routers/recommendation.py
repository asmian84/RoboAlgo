"""
RoboAlgo - Trade Recommendation Engine
Combines MTF confidence, XGBoost daily signal, pattern context,
and behavioral sentiment to produce a consolidated trade recommendation.
"""

from datetime import date, timedelta

from fastapi import APIRouter
from sqlalchemy import desc, select

import numpy as np

from api.routers.alphavantage import get_earnings_date, get_news_sentiment_score
from api.routers.mtf import TIMEFRAMES, get_mtf, _get_underlying
from database.connection import get_session
from database.models import Feature, Indicator, Instrument, PatternDetection, PriceData, Signal, VolatilityRegime
from config.settings import VOL_LOW, VOL_NORMAL, VOL_HIGH

router = APIRouter()

# Timeframe weights: longer = more weight
_TF_WEIGHTS = {
    "1m": 0.1,  "5m": 0.25,
    "15m": 0.5, "30m": 0.75, "1h": 1.0, "2h": 1.25,
    "4h": 1.5,  "1d": 2.5,   "1w": 4.0, "1M": 5.0,
}

# Phase quality for conviction scoring (0–100)
_PHASE_QUALITY = {
    "Accumulation":  100,
    "Early Bull":     95,
    "Recovery":       80,
    "Momentum Bull":  70,
    "Late Bull":      55,
    "Distribution":   20,
    "Early Bear":     12,
    "Late Bear":       8,
    "Markdown":        5,
    "Capitulation":   10,
}


def _horizon_plan(tf_entries: list, current_price: float) -> dict | None:
    """BUY/ADD/SCALE/EXIT levels scaled to current_price from underlying BB/ATR percentages."""
    valid = [e for e in tf_entries if e.get("details", {}).get("atr") is not None]
    if not valid or not current_price:
        return None

    def _avg(key: str):
        vals = [e["details"][key] for e in valid if e["details"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    atr_raw   = _avg("atr")
    ref_price = _avg("last_price")
    if not atr_raw or atr_raw <= 0 or not ref_price or ref_price <= 0:
        return None

    # Express everything as % of reference price, then re-scale to actual symbol price
    atr = current_price * (atr_raw / ref_price)

    def _scale(raw):
        return current_price * (raw / ref_price) if raw is not None else None

    bb_lower = _scale(_avg("bb_lower"))
    bb_mid   = _scale(_avg("bb_mid"))
    bb_upper = _scale(_avg("bb_upper"))

    buy   = bb_lower if bb_lower else current_price - atr * 1.0
    add   = buy - atr * 0.75
    scale = bb_mid   if bb_mid   else current_price + atr * 1.5
    exit_ = bb_upper if bb_upper else current_price + atr * 3.0

    # Ensure correct ordering
    buy   = max(buy,   add   + atr * 0.1)
    scale = max(scale, buy   + atr * 0.5)
    exit_ = max(exit_, scale + atr * 0.5)

    return {
        "buy_price":        round(buy,   2),
        "accumulate_price": round(add,   2),
        "scale_price":      round(scale, 2),
        "sell_price":       round(exit_, 2),
    }


def _recommendation_label(score: float) -> tuple[str, str]:
    if score >= 75:
        return "STRONG BUY", "#22c55e"
    elif score >= 62:
        return "BUY", "#86efac"
    elif score >= 48:
        return "NEUTRAL", "#eab308"
    elif score >= 35:
        return "WAIT", "#f97316"
    else:
        return "AVOID", "#ef4444"


def _conviction_label(composite: float) -> tuple[str, str]:
    """Composite conviction: 60% probability + 40% phase quality."""
    if composite >= 78:
        return "HIGH", "#22c55e"
    elif composite >= 60:
        return "MEDIUM", "#eab308"
    else:
        return "LOW", "#ef4444"


def _behavioral_signal(ind: Indicator | None, feat: Feature | None,
                        bull_patterns: int, bear_patterns: int,
                        sentiment_score: float | None = None) -> dict:
    """
    Detect human behavioral patterns from indicator data + news sentiment.
    Fear/capitulation = contrarian buy. FOMO/euphoria = caution.
    sentiment_score: Alpha Vantage ticker sentiment (-1 to +1), optional.
    """
    if not ind:
        return {
            "signal": "NEUTRAL", "label": "No indicator data",
            "description": "Cannot assess behavioral sentiment.",
            "strength": 50, "action": "FOLLOW SIGNAL", "color": "#6b7280",
        }

    rsi         = ind.rsi or 50.0
    bb_pos      = feat.bb_position if feat else 0.5     # 0=lower band, 1=upper band
    vol_ratio   = feat.volume_ratio if feat else 1.0    # >1 = above avg volume
    ret_5d      = feat.return_5d if feat else 0.0       # 5-day return
    momentum    = feat.momentum if feat else 0.0

    # ── Fear / capitulation score ────────────────────────────────────────────
    fear = 0.0
    if rsi < 25:       fear += 45
    elif rsi < 30:     fear += 35
    elif rsi < 40:     fear += 20

    if bb_pos is not None:
        if bb_pos < 0.10:  fear += 35
        elif bb_pos < 0.20: fear += 25
        elif bb_pos < 0.30: fear += 12

    if ret_5d is not None:
        if ret_5d < -0.10:  fear += 25
        elif ret_5d < -0.05: fear += 15
        elif ret_5d < -0.02: fear += 7

    if vol_ratio is not None and vol_ratio > 1.5 and ret_5d is not None and ret_5d < 0:
        fear += 15  # high volume sell-off = capitulation signal

    if bull_patterns > bear_patterns:
        fear += 10  # bullish patterns forming during fear = stronger signal

    # ── Greed / FOMO score ───────────────────────────────────────────────────
    greed = 0.0
    if rsi > 75:       greed += 45
    elif rsi > 70:     greed += 35
    elif rsi > 60:     greed += 15

    if bb_pos is not None:
        if bb_pos > 0.90:  greed += 35
        elif bb_pos > 0.80: greed += 25
        elif bb_pos > 0.70: greed += 12

    if ret_5d is not None:
        if ret_5d > 0.10:  greed += 25
        elif ret_5d > 0.05: greed += 15
        elif ret_5d > 0.02: greed += 7

    if vol_ratio is not None and vol_ratio > 2.0 and ret_5d is not None and ret_5d > 0:
        greed += 15  # parabolic volume surge = FOMO

    # ── News sentiment modifier (Alpha Vantage) ───────────────────────────────
    if sentiment_score is not None:
        if sentiment_score < -0.35:
            fear  += 20   # strongly negative news amplifies fear reading
        elif sentiment_score < -0.15:
            fear  += 10
        elif sentiment_score > 0.35:
            greed += 15   # strongly positive news amplifies optimism
        elif sentiment_score > 0.15:
            greed += 8

    # ── Classify ─────────────────────────────────────────────────────────────
    fear  = min(fear, 100)
    greed = min(greed, 100)

    if fear >= 55:
        return {
            "signal": "FEAR_CAPITULATION",
            "label": "Fear / Capitulation",
            "description": "Retail panic-selling. Smart money accumulates on weakness.",
            "strength": round(fear),
            "action": "CONTRARIAN BUY",
            "color": "#22c55e",
        }
    elif greed >= 55:
        return {
            "signal": "GREED_FOMO",
            "label": "FOMO / Euphoria",
            "description": "Retail chasing parabolic moves. Risk of reversal is elevated.",
            "strength": round(greed),
            "action": "TAKE PROFIT / REDUCE",
            "color": "#ef4444",
        }
    elif fear >= 35:
        return {
            "signal": "MILD_FEAR",
            "label": "Mild Fear",
            "description": "Market is cautious. Watch for a bullish reversal confirmation.",
            "strength": round(fear),
            "action": "WATCH — NEAR ENTRY",
            "color": "#f97316",
        }
    elif greed >= 35:
        return {
            "signal": "MILD_GREED",
            "label": "Mild Optimism",
            "description": "Market stretched but not extreme. Trail stops on open positions.",
            "strength": round(greed),
            "action": "TRAIL STOPS",
            "color": "#fbbf24",
        }
    else:
        return {
            "signal": "NEUTRAL",
            "label": "Neutral Sentiment",
            "description": "No extreme behavioral signal. Follow the model.",
            "strength": 50,
            "action": "FOLLOW SIGNAL",
            "color": "#6b7280",
        }


_BENCHMARK_SYMBOLS = ["SPY", "QQQ"]  # Primary benchmarks for correlation


def _compute_index_correlation(session, symbol: str, lookback: int = 20) -> dict:
    """
    Compute 20-day rolling return correlation between symbol and SPY/QQQ.
    Returns label (CORRELATED / DECOUPLED / INVERSE / INSUFFICIENT_DATA)
    and numeric correlation for the primary benchmark.
    """
    result = {"correlation": None, "benchmark": None, "label": "INSUFFICIENT DATA", "color": "#6b7280"}

    # Load symbol prices
    instr = session.execute(
        select(Instrument).where(Instrument.symbol == symbol)
    ).scalar_one_or_none()
    if not instr:
        return result

    sym_prices = list(session.execute(
        select(PriceData.date, PriceData.close)
        .where(PriceData.instrument_id == instr.id)
        .order_by(PriceData.date.desc())
        .limit(lookback + 5)
    ).all())
    if len(sym_prices) < lookback:
        return result

    sym_prices = sorted(sym_prices, key=lambda r: r.date)
    sym_closes = [float(r.close) for r in sym_prices if r.close]
    if len(sym_closes) < lookback:
        return result

    best_corr: float | None = None
    best_bench: str | None = None

    for bench in _BENCHMARK_SYMBOLS:
        bench_instr = session.execute(
            select(Instrument).where(Instrument.symbol == bench)
        ).scalar_one_or_none()
        if not bench_instr:
            continue

        bench_prices = list(session.execute(
            select(PriceData.date, PriceData.close)
            .where(PriceData.instrument_id == bench_instr.id)
            .order_by(PriceData.date.desc())
            .limit(lookback + 5)
        ).all())
        if len(bench_prices) < lookback:
            continue

        bench_prices = sorted(bench_prices, key=lambda r: r.date)
        bench_closes = [float(r.close) for r in bench_prices if r.close]

        # Align by date
        n = min(len(sym_closes), len(bench_closes), lookback + 1)
        if n < 5:
            continue

        s_arr = np.array(sym_closes[-n:])
        b_arr = np.array(bench_closes[-n:])
        s_ret = np.diff(s_arr) / s_arr[:-1]
        b_ret = np.diff(b_arr) / b_arr[:-1]

        if len(s_ret) < 4:
            continue

        try:
            corr = float(np.corrcoef(s_ret, b_ret)[0, 1])
            if np.isnan(corr):
                continue
        except Exception:
            continue

        if best_corr is None or abs(corr) > abs(best_corr):
            best_corr = corr
            best_bench = bench

    if best_corr is None:
        return result

    # Classify
    if best_corr >= 0.70:
        label, color = "CORRELATED", "#9ca3af"
        description = f"Moves with {best_bench} ({best_corr:.2f}). Limited alpha — index exposure."
    elif best_corr <= -0.50:
        label, color = "INVERSE", "#ef4444"
        description = f"Moves opposite to {best_bench} ({best_corr:.2f}). Bear ETF or hedge."
    elif abs(best_corr) < 0.30:
        label, color = "DECOUPLED", "#22c55e"
        description = f"Low correlation to {best_bench} ({best_corr:.2f}). Alpha stock — independent catalyst."
    else:
        label, color = "SEMI-CORRELATED", "#eab308"
        description = f"Moderate correlation to {best_bench} ({best_corr:.2f}). Follows index trend loosely."

    return {
        "correlation": round(best_corr, 3),
        "benchmark":   best_bench,
        "label":       label,
        "color":       color,
        "description": description,
    }


@router.get("/{symbol}")
def get_recommendation(symbol: str):
    """Consolidated trade recommendation combining MTF, XGBoost, patterns, and behavior."""
    symbol = symbol.upper()
    underlying = _get_underlying(symbol)
    session = get_session()
    try:
        # ── Latest XGBoost signal ─────────────────────────────────────────────
        sig = session.execute(
            select(Signal)
            .join(Instrument, Signal.instrument_id == Instrument.id)
            .where(Instrument.symbol == symbol)
            .order_by(desc(Signal.date))
            .limit(1)
        ).scalar_one_or_none()

        xgb_prob = float(sig.probability) * 100.0 if sig else 50.0
        market_phase = sig.market_phase if sig else "Unknown"

        # ── Recent patterns (last 14 days) ────────────────────────────────────
        since = date.today() - timedelta(days=14)
        instr = session.execute(
            select(Instrument).where(Instrument.symbol == symbol)
        ).scalar_one_or_none()

        patterns = []
        if instr:
            patterns = list(session.execute(
                select(PatternDetection)
                .where(PatternDetection.instrument_id == instr.id)
                .where(PatternDetection.date >= since)
                .order_by(desc(PatternDetection.date))
            ).scalars())

        bull_count = sum(1 for p in patterns if p.direction == "bullish")
        bear_count = sum(1 for p in patterns if p.direction == "bearish")
        pat_score  = 50.0 + (bull_count - bear_count) / max(len(patterns), 1) * 30.0
        pat_score  = max(0.0, min(100.0, pat_score))

        # ── Latest indicators + features (for behavioral signal) ──────────────
        latest_ind, latest_feat = None, None
        if instr:
            latest_ind = session.execute(
                select(Indicator)
                .where(Indicator.instrument_id == instr.id)
                .order_by(desc(Indicator.date))
                .limit(1)
            ).scalar_one_or_none()

            latest_feat = session.execute(
                select(Feature)
                .where(Feature.instrument_id == instr.id)
                .order_by(desc(Feature.date))
                .limit(1)
            ).scalar_one_or_none()

        # ── Index correlation ─────────────────────────────────────────────────
        index_corr = _compute_index_correlation(session, underlying or symbol)

        # ── Alpha Vantage: news sentiment + earnings risk ─────────────────────
        # Run synchronously (cached after first call, so no latency after warm-up)
        av_sentiment_score, av_article_count = get_news_sentiment_score(symbol)
        av_earnings_date = get_earnings_date(symbol)

        earnings_days_until = (
            (av_earnings_date - date.today()).days if av_earnings_date else None
        )
        earnings_risk = earnings_days_until is not None and earnings_days_until <= 5

        behavioral = _behavioral_signal(
            latest_ind, latest_feat, bull_count, bear_count,
            sentiment_score=av_sentiment_score,
        )

        # ── Volatility regime gate ─────────────────────────────────────────────
        vol_regime_row = None
        if instr:
            vol_regime_row = session.execute(
                select(VolatilityRegime)
                .where(VolatilityRegime.instrument_id == instr.id)
                .order_by(desc(VolatilityRegime.date))
                .limit(1)
            ).scalar_one_or_none()

        _vol_regime     = vol_regime_row.regime if vol_regime_row else VOL_NORMAL
        _is_compression = bool(vol_regime_row.is_compression) if vol_regime_row else False
        _is_expansion   = bool(vol_regime_row.is_expansion)   if vol_regime_row else False
        _signal_gated   = (_vol_regime == VOL_LOW)   # suppress all signals in LOW_VOL

        # Regime label and color for display
        _regime_color = {
            VOL_LOW:    "#6b7280",  # grey  — no trading
            VOL_NORMAL: "#f59e0b",  # amber — limited
            VOL_HIGH:   "#10b981",  # green — active
        }.get(_vol_regime, "#9ca3af")

        # ── MTF analysis ──────────────────────────────────────────────────────
        mtf_data = get_mtf(symbol)
        all_tfs  = mtf_data["timeframes"]
        valid    = [tf for tf in all_tfs if tf["confidence"] is not None]

        if valid:
            w_sum = sum(tf["confidence"] * _TF_WEIGHTS.get(tf["timeframe"], 1.0) for tf in valid)
            w_tot = sum(_TF_WEIGHTS.get(tf["timeframe"], 1.0) for tf in valid)
            mtf_avg   = w_sum / w_tot
            alignment = sum(1 for tf in valid if tf["signal"] == "bullish") / len(valid)
        else:
            mtf_avg   = 50.0
            alignment = 0.5

        # ── Horizon-specific trade plans ──────────────────────────────────────
        _tf_map = {e["timeframe"]: e for e in all_tfs}
        # Use actual symbol's latest close from DB (not underlying) for correct price scaling
        _price_row = session.execute(
            select(PriceData)
            .join(Instrument, PriceData.instrument_id == Instrument.id)
            .where(Instrument.symbol == symbol)
            .order_by(desc(PriceData.date))
            .limit(1)
        ).scalar_one_or_none()
        _cur_price = float(_price_row.close) if _price_row else None
        horizon_plans = {
            "short":  _horizon_plan([_tf_map[t] for t in ["1m","5m","15m","30m"]  if t in _tf_map], _cur_price),
            "medium": _horizon_plan([_tf_map[t] for t in ["1h","2h","4h"]         if t in _tf_map], _cur_price),
            "long":   _horizon_plan([_tf_map[t] for t in ["1d","1w","1M"]         if t in _tf_map], _cur_price),
        }

        # ── Top-down alignment: trend on higher TFs, entry on lower TFs ─────────
        _higher_tfs  = [_tf_map[t] for t in ["1d", "1w", "1M"] if t in _tf_map and _tf_map[t]["confidence"] is not None]
        _lower_tfs   = [_tf_map[t] for t in ["1m", "5m", "15m", "30m", "1h", "2h", "4h"] if t in _tf_map and _tf_map[t]["confidence"] is not None]

        _higher_bull = sum(1 for e in _higher_tfs if e["signal"] == "bullish")
        _higher_bear = sum(1 for e in _higher_tfs if e["signal"] == "bearish")
        _higher_conf = sum(e["confidence"] for e in _higher_tfs) / len(_higher_tfs) if _higher_tfs else 50.0

        if _higher_bull >= 2:
            _higher_bias = "bullish"
        elif _higher_bear >= 2:
            _higher_bias = "bearish"
        else:
            _higher_bias = "neutral"

        # Pullback quality: lower TF pulled back = lower confidence = better entry in uptrend
        _lower_conf      = sum(e["confidence"] for e in _lower_tfs) / len(_lower_tfs) if _lower_tfs else 50.0
        _pullback_score  = round(100.0 - _lower_conf, 1)   # 0=no pullback, 100=fully oversold

        # Combined top-down signal
        if _higher_bias == "bullish" and _pullback_score >= 50:
            _td_signal, _td_color = "ENTER", "#10b981"
        elif _higher_bias == "bullish" and _pullback_score >= 30:
            _td_signal, _td_color = "WATCH", "#f59e0b"
        elif _higher_bias == "bullish":
            _td_signal, _td_color = "WAIT — EXTENDED", "#6b7280"
        elif _higher_bias == "bearish":
            _td_signal, _td_color = "AVOID", "#ef4444"
        else:
            _td_signal, _td_color = "NEUTRAL", "#a78bfa"

        top_down = {
            "higher_bias":    _higher_bias,
            "higher_score":   round(_higher_conf, 1),
            "higher_aligned": _higher_bias == "bullish",
            "pullback_score": _pullback_score,
            "signal":         _td_signal,
            "signal_color":   _td_color,
        }

        # ── Overall recommendation score ──────────────────────────────────────
        overall = (
            0.35 * mtf_avg +
            0.35 * xgb_prob +
            0.20 * (alignment * 100.0) +
            0.10 * pat_score
        )
        overall = round(overall, 1)
        label, color = _recommendation_label(overall)

        # ── Volatility gate: suppress signals in LOW_VOL ──────────────────────
        if _signal_gated:
            label  = "WAIT — LOW VOL"
            color  = "#6b7280"
            overall = min(overall, 40.0)   # Cap below BUY threshold

        # ── Conviction score (composite: prob + phase quality) ────────────────
        phase_quality   = _PHASE_QUALITY.get(market_phase, 50)
        conviction_raw  = 0.60 * xgb_prob + 0.40 * phase_quality
        conviction_label, conviction_color = _conviction_label(conviction_raw)

        # ── 3TT trade plan ────────────────────────────────────────────────────
        trade_plan = None
        if sig:
            trade_plan = {
                "date":             sig.date.isoformat(),
                "confidence_tier":  sig.confidence_tier,
                "conviction":       conviction_label,
                "conviction_score": round(conviction_raw, 1),
                "market_phase":     sig.market_phase,
                "buy_price":        sig.buy_price,
                "accumulate_price": sig.accumulate_price,
                "scale_price":      sig.scale_price,
                "sell_price":       sig.sell_price,
            }

        # ── Sentiment label for display ────────────────────────────────────────
        if av_sentiment_score is not None:
            if av_sentiment_score > 0.25:
                av_sentiment_label, av_sentiment_color = "BULLISH", "#22c55e"
            elif av_sentiment_score < -0.25:
                av_sentiment_label, av_sentiment_color = "BEARISH", "#ef4444"
            else:
                av_sentiment_label, av_sentiment_color = "NEUTRAL", "#9ca3af"
        else:
            av_sentiment_label, av_sentiment_color = "N/A", "#6b7280"

        return {
            "symbol":               symbol,
            "underlying":           underlying,
            "recommendation":       label,
            "recommendation_color": color,
            "overall_score":        overall,
            "conviction":           conviction_label,
            "conviction_color":     conviction_color,
            "conviction_score":     round(conviction_raw, 1),
            "components": {
                "mtf_weighted_avg":    round(mtf_avg, 1),
                "xgb_probability":     round(xgb_prob, 1),
                "tf_alignment_pct":    round(alignment * 100.0, 1),
                "pattern_score":       round(pat_score, 1),
                "phase_quality":       phase_quality,
                "news_sentiment_score": round(av_sentiment_score, 3) if av_sentiment_score is not None else None,
            },
            "mtf_timeframes":  all_tfs,
            "horizon_plans":   horizon_plans,
            "top_down":        top_down,
            "pattern_context": {
                "bullish": bull_count,
                "bearish": bear_count,
                "total":   len(patterns),
                "recent":  [
                    {"date": p.date.isoformat(), "name": p.pattern_name,
                     "direction": p.direction, "strength": p.strength}
                    for p in patterns[:10]
                ],
            },
            "behavioral": behavioral,
            "trade_plan": trade_plan,
            "news_sentiment": {
                "score":         round(av_sentiment_score, 3) if av_sentiment_score is not None else None,
                "article_count": av_article_count,
                "label":         av_sentiment_label,
                "color":         av_sentiment_color,
            },
            "earnings_risk": {
                "has_risk":      earnings_risk,
                "earnings_date": av_earnings_date.isoformat() if av_earnings_date else None,
                "days_until":    earnings_days_until,
            },
            "index_correlation": index_corr,
            "volatility_regime": {
                "regime":                _vol_regime,
                "regime_color":          _regime_color,
                "is_compression":        _is_compression,
                "is_expansion":          _is_expansion,
                "signal_gated":          _signal_gated,
                "atr_percentile":        round(vol_regime_row.atr_percentile, 3) if vol_regime_row and vol_regime_row.atr_percentile is not None else None,
                "bb_width_percentile":   round(vol_regime_row.bb_width_percentile, 3) if vol_regime_row and vol_regime_row.bb_width_percentile is not None else None,
                "realized_vol_20d":      round(vol_regime_row.realized_vol_20d, 4) if vol_regime_row and vol_regime_row.realized_vol_20d is not None else None,
                "compression_range_high": vol_regime_row.compression_range_high if vol_regime_row else None,
                "compression_range_low":  vol_regime_row.compression_range_low  if vol_regime_row else None,
                "gate_reason":           "LOW_VOL: no signals — wait for volatility to expand." if _signal_gated else None,
                "setup_quality":         "COMPRESSION — breakout imminent" if _is_compression and not _is_expansion
                                         else "EXPANSION — breakout in progress" if _is_expansion
                                         else "NORMAL — standard conditions",
            },
        }
    finally:
        session.close()
