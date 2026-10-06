"""Pattern endpoints — algorithmic pattern states + signal-ready output."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from fastapi import APIRouter, Query

from pattern_engine.service import PatternService

router = APIRouter()
service = PatternService()

# ── Scan cache — one cache slot per timeframe ──────────────────────────────
# Key: tf string ("daily" | "1h" | "4h" etc.)  Value: (results, timestamp)
_scan_cache: dict[str, tuple[list, float]] = {}
_SCAN_TTL: float = 300.0  # 5 minutes per TF slot

# ── Elliott Wave result cache — avoid re-running expensive detection ──────
# Key: "SYMBOL:tf:bars"  Value: (response_dict, timestamp)
_ew_cache: dict[str, tuple[dict, float]] = {}
_EW_CACHE_TTL: float = 300.0  # 5 min — wave counts don't change intraday

# Map URL timeframe strings to resolution_minutes (0 = daily from DB)
_TF_RESOLUTION: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "60m": 60,
    "2h": 120, "4h": 240,
}

SUPPORTED_PATTERNS = [
    # ── Chart patterns (18 sub-patterns) ──────────────────────────────────
    {"pattern_name": "Chair Pattern",        "type": "chart", "direction": "bullish",  "description": "3-phase: impulse leg, sideways compression (seat), recovery breakout above neckline."},
    {"pattern_name": "Cup & Handle",         "type": "chart", "direction": "bullish",  "description": "U-shaped cup arc with shallow handle pullback; breakout above rim = bullish."},
    {"pattern_name": "Bull Flag",            "type": "chart", "direction": "bullish",  "description": "Sharp upward pole followed by a downward-drifting parallel channel consolidation."},
    {"pattern_name": "Bear Flag",            "type": "chart", "direction": "bearish",  "description": "Sharp downward pole followed by an upward-drifting parallel channel consolidation."},
    {"pattern_name": "Bullish Pennant",      "type": "chart", "direction": "bullish",  "description": "Upward pole + short converging symmetrical triangle; breakout above resistance."},
    {"pattern_name": "Bearish Pennant",      "type": "chart", "direction": "bearish",  "description": "Downward pole + short converging symmetrical triangle; breakdown below support."},
    {"pattern_name": "Ascending Channel",    "type": "chart", "direction": "bullish",  "description": "Parallel rising trendlines; breakout above upper channel = acceleration."},
    {"pattern_name": "Descending Channel",   "type": "chart", "direction": "bearish",  "description": "Parallel falling trendlines; breakout above upper channel = reversal."},
    {"pattern_name": "Symmetrical Triangle", "type": "chart", "direction": "neutral",  "description": "Converging highs/lows — compression before a directional breakout."},
    {"pattern_name": "Ascending Triangle",   "type": "chart", "direction": "bullish",  "description": "Flat resistance + rising support; bullish breakout above resistance."},
    {"pattern_name": "Descending Triangle",  "type": "chart", "direction": "bearish",  "description": "Falling resistance + flat support; bearish breakdown below support."},
    {"pattern_name": "Rising Wedge",         "type": "chart", "direction": "bearish",  "description": "Both trendlines rising with support steeper; price squeezed upward — bearish."},
    {"pattern_name": "Falling Wedge",        "type": "chart", "direction": "bullish",  "description": "Both trendlines falling with resistance steeper; price squeezed downward — bullish."},
    {"pattern_name": "Head & Shoulders",     "type": "chart", "direction": "bearish",  "description": "Left shoulder, higher head, right shoulder; breakdown below neckline = bearish."},
    {"pattern_name": "Inv. Head & Shoulders","type": "chart", "direction": "bullish",  "description": "Inverse: three troughs; breakout above neckline = bullish reversal."},
    {"pattern_name": "Double Top",           "type": "chart", "direction": "bearish",  "description": "Two roughly equal swing highs; breakdown below neckline = bearish reversal."},
    {"pattern_name": "Double Bottom",        "type": "chart", "direction": "bullish",  "description": "Two roughly equal swing lows; breakout above neckline = bullish reversal."},
    {"pattern_name": "Triple Top",           "type": "chart", "direction": "bearish",  "description": "Three near-equal highs; more conviction than double top; breakdown bearish."},
    {"pattern_name": "Triple Bottom",        "type": "chart", "direction": "bullish",  "description": "Three near-equal lows; more conviction than double bottom; breakout bullish."},
    {"pattern_name": "Rounding Bottom",      "type": "chart", "direction": "bullish",  "description": "Saucer / bowl — gradual U-shaped deceleration then recovery; breakout bullish."},
    {"pattern_name": "Rounding Top",         "type": "chart", "direction": "bearish",  "description": "Dome / n-shape — gradual distribution; breakdown bearish."},
    {"pattern_name": "Rectangle",            "type": "chart", "direction": "neutral",  "description": "Price consolidates between flat resistance and support; breakout determines direction."},
    {"pattern_name": "Megaphone",            "type": "chart", "direction": "neutral",  "description": "Broadening formation — rising highs + falling lows; high volatility expansion."},
    # ── Harmonic patterns ─────────────────────────────────────────────────
    {"pattern_name": "Harmonic Pattern",     "type": "harmonic", "direction": "neutral", "description": "XABCD Fibonacci ratio patterns: Gartley, Bat, Butterfly, Crab, Cypher."},
    # ── Gann ──────────────────────────────────────────────────────────────
    {"pattern_name": "Gann Levels",          "type": "gann",     "direction": "neutral", "description": "9-angle fan projection (8x1 through 1x8), retracement levels (1/8-7/8), time cycles."},
    # ── Wyckoff ───────────────────────────────────────────────────────────
    {"pattern_name": "Wyckoff Structure",    "type": "wyckoff",  "direction": "neutral", "description": "Accumulation + Distribution structural phases (A-E) with event labeling."},
    # ── Candlestick patterns ───────────────────────────────────────────────
    {"pattern_name": "Doji",               "type": "candlestick", "direction": "neutral", "description": "Open ≈ Close, indicating indecision. Signals potential reversal."},
    {"pattern_name": "Hammer",             "type": "candlestick", "direction": "bullish", "description": "Small body at top, long lower shadow. Bullish reversal after downtrend."},
    {"pattern_name": "Inverted Hammer",    "type": "candlestick", "direction": "bullish", "description": "Small body at bottom, long upper shadow. Potential bullish reversal."},
    {"pattern_name": "Hanging Man",        "type": "candlestick", "direction": "bearish", "description": "Small body at top, long lower shadow in uptrend. Bearish reversal warning."},
    {"pattern_name": "Shooting Star",      "type": "candlestick", "direction": "bearish", "description": "Small body at bottom, long upper shadow in uptrend. Bearish reversal."},
    {"pattern_name": "Bullish Engulfing",  "type": "candlestick", "direction": "bullish", "description": "Large green candle fully engulfs prior red candle. Strong bullish reversal."},
    {"pattern_name": "Bearish Engulfing",  "type": "candlestick", "direction": "bearish", "description": "Large red candle fully engulfs prior green candle. Strong bearish reversal."},
    {"pattern_name": "Morning Star",       "type": "candlestick", "direction": "bullish", "description": "3-candle bullish reversal: large red → small doji → large green."},
    {"pattern_name": "Evening Star",       "type": "candlestick", "direction": "bearish", "description": "3-candle bearish reversal: large green → small doji → large red."},
    {"pattern_name": "Three White Soldiers","type": "candlestick", "direction": "bullish", "description": "Three consecutive large green candles. Strong bullish continuation."},
    {"pattern_name": "Three Black Crows",  "type": "candlestick", "direction": "bearish", "description": "Three consecutive large red candles. Strong bearish continuation."},
    {"pattern_name": "Bullish Harami",     "type": "candlestick", "direction": "bullish", "description": "Small green inside prior red. Bullish reversal hint."},
    {"pattern_name": "Bearish Harami",     "type": "candlestick", "direction": "bearish", "description": "Small red inside prior green. Bearish reversal hint."},
    {"pattern_name": "Piercing Line",      "type": "candlestick", "direction": "bullish", "description": "Green candle closes above prior red midpoint. Bullish."},
    {"pattern_name": "Dark Cloud Cover",   "type": "candlestick", "direction": "bearish", "description": "Red candle closes below prior green midpoint. Bearish."},
    {"pattern_name": "Bullish Marubozu",   "type": "candlestick", "direction": "bullish", "description": "Full green candle with tiny wicks. Strong buying pressure."},
    {"pattern_name": "Bearish Marubozu",   "type": "candlestick", "direction": "bearish", "description": "Full red candle with tiny wicks. Strong selling pressure."},
    # ── Behavioral patterns ────────────────────────────────────────────────
    {"pattern_name": "Banister",            "type": "behavioral", "direction": "bullish", "description": "Descending staircase of lower highs/lows then breaks above last swing high."},
    {"pattern_name": "Capitulation Bottom", "type": "behavioral", "direction": "bullish", "description": "Rapid drop with volume spike followed by green reversal candle."},
    {"pattern_name": "Wave 4 Pullback",     "type": "behavioral", "direction": "bullish", "description": "After strong advance, 8-15% pullback to MA50 with RSI reset."},
    {"pattern_name": "ABC Correction",      "type": "behavioral", "direction": "bullish", "description": "Three-wave corrective decline creating C-wave reversal opportunity."},
    {"pattern_name": "Staircase Uptrend",   "type": "behavioral", "direction": "bullish", "description": "3+ ascending higher lows — buy the step."},
    {"pattern_name": "Herd Exhaustion",     "type": "behavioral", "direction": "bearish", "description": "RSI>75 + above BB upper + declining volume. FOMO top."},
    {"pattern_name": "Dead Cat Bounce",     "type": "behavioral", "direction": "bearish", "description": "Weak bounce after sell-off on low volume. Continuation down."},
    {"pattern_name": "FOMO Breakout",       "type": "behavioral", "direction": "bullish", "description": "New 20-day high with volume spike + healthy RSI."},
    {"pattern_name": "V-Bottom Reversal",   "type": "behavioral", "direction": "bullish", "description": "Sharp flush with panic selling then strong recovery."},
    {"pattern_name": "Distribution Top",    "type": "behavioral", "direction": "bearish", "description": "High-volume selling near 20-day high + RSI divergence."},
    # ── Indicator patterns ─────────────────────────────────────────────────
    {"pattern_name": "RSI Regular Bullish Divergence",  "type": "indicator", "direction": "bullish", "description": "Price makes lower low while RSI makes higher low. Bullish reversal signal."},
    {"pattern_name": "RSI Regular Bearish Divergence",  "type": "indicator", "direction": "bearish", "description": "Price makes higher high while RSI makes lower high. Bearish reversal signal."},
    {"pattern_name": "RSI Hidden Bullish Divergence",   "type": "indicator", "direction": "bullish", "description": "Price makes higher low while RSI makes lower low. Bullish continuation."},
    {"pattern_name": "RSI Hidden Bearish Divergence",   "type": "indicator", "direction": "bearish", "description": "Price makes lower high while RSI makes higher high. Bearish continuation."},
    {"pattern_name": "MACD Bullish Crossover",          "type": "indicator", "direction": "bullish", "description": "MACD line crosses above signal line. Bullish momentum."},
    {"pattern_name": "MACD Bearish Crossover",          "type": "indicator", "direction": "bearish", "description": "MACD line crosses below signal line. Bearish momentum."},
    {"pattern_name": "MACD Zero Line Cross Up",         "type": "indicator", "direction": "bullish", "description": "MACD crosses above zero. Strong bullish confirmation."},
    {"pattern_name": "MACD Zero Line Cross Down",       "type": "indicator", "direction": "bearish", "description": "MACD crosses below zero. Strong bearish confirmation."},
    {"pattern_name": "MACD Histogram Divergence Bull",  "type": "indicator", "direction": "bullish", "description": "Price lower low but MACD histogram higher low. Bullish."},
    {"pattern_name": "MACD Histogram Divergence Bear",  "type": "indicator", "direction": "bearish", "description": "Price higher high but MACD histogram lower high. Bearish."},
    # ── Volume patterns ────────────────────────────────────────────────────
    {"pattern_name": "Buying Climax",           "type": "volume", "direction": "bearish", "description": "Extreme volume spike on a sharp up move. Distribution signal."},
    {"pattern_name": "Selling Climax",          "type": "volume", "direction": "bullish", "description": "Extreme volume spike on a sharp down move. Exhaustion reversal."},
    {"pattern_name": "Volume Dry Up",           "type": "volume", "direction": "neutral",  "description": "3+ bars of very low volume. Coiling before big move."},
    {"pattern_name": "OBV Bullish Divergence",  "type": "volume", "direction": "bullish", "description": "Price makes lower low but OBV makes higher low. Smart money accumulating."},
    {"pattern_name": "OBV Bearish Divergence",  "type": "volume", "direction": "bearish", "description": "Price makes higher high but OBV makes lower high. Smart money distributing."},
    {"pattern_name": "Institutional Accumulation", "type": "volume", "direction": "bullish", "description": "More accumulation days than distribution days. Institutional buying."},
    {"pattern_name": "Institutional Distribution", "type": "volume", "direction": "bearish", "description": "More distribution days than accumulation days. Institutional selling."},
    {"pattern_name": "HVN Support Bounce",      "type": "volume", "direction": "bullish", "description": "Price bounces off high-volume node (key support level)."},
    {"pattern_name": "HVN Resistance Rejection","type": "volume", "direction": "bearish", "description": "Price rejected at high-volume node (key resistance level)."},
    # ── Measured move patterns ─────────────────────────────────────────────
    {"pattern_name": "Bullish Measured Move",    "type": "measured_move", "direction": "bullish", "description": "AB=CD pattern: equal legs bullish projection from C pivot."},
    {"pattern_name": "Bearish Measured Move",    "type": "measured_move", "direction": "bearish", "description": "AB=CD pattern: equal legs bearish projection from C pivot."},
    {"pattern_name": "Bullish Extension",        "type": "measured_move", "direction": "bullish", "description": "Measured move extension target beyond initial projection."},
    {"pattern_name": "Bearish Extension",        "type": "measured_move", "direction": "bearish", "description": "Measured move extension target beyond initial projection."},
    # ── Strategy patterns ─────────────────────────────────────────────────
    {"pattern_name": "BB Squeeze Breakout",              "type": "strategy", "direction": "neutral",  "description": "Bollinger Bands inside Keltner Channels, then release. Volatility explosion."},
    {"pattern_name": "RSI(2) Oversold Setup",            "type": "strategy", "direction": "bullish",  "description": "2-period RSI below 10 while above 200 MA. Short-term mean reversion buy."},
    {"pattern_name": "RSI(2) Overbought Exit",           "type": "strategy", "direction": "bearish",  "description": "2-period RSI above 90. Short-term overbought — exit longs."},
    {"pattern_name": "Stochastic Pop",                   "type": "strategy", "direction": "bullish",  "description": "Stochastic crosses above 80. Momentum breakout continuation."},
    {"pattern_name": "Stochastic Drop",                  "type": "strategy", "direction": "bearish",  "description": "Stochastic crosses below 20. Bearish momentum breakdown."},
    {"pattern_name": "CCI Correction Bullish",           "type": "strategy", "direction": "bullish",  "description": "CCI crosses above -100 after oversold. Bullish correction entry."},
    {"pattern_name": "CCI Correction Bearish",           "type": "strategy", "direction": "bearish",  "description": "CCI crosses below +100 after overbought. Bearish correction entry."},
    {"pattern_name": "Golden Cross Setup",               "type": "strategy", "direction": "bullish",  "description": "MA50 just crossed above MA200. Long-term trend confirmation."},
    {"pattern_name": "Death Cross Setup",                "type": "strategy", "direction": "bearish",  "description": "MA50 just crossed below MA200. Long-term bearish confirmation."},
    {"pattern_name": "MA50 Support Bounce",              "type": "strategy", "direction": "bullish",  "description": "Price bounced off 50-day moving average."},
    {"pattern_name": "MA200 Support Bounce",             "type": "strategy", "direction": "bullish",  "description": "Price bounced off 200-day moving average. Major support."},
    {"pattern_name": "MA50 Resistance Rejection",        "type": "strategy", "direction": "bearish",  "description": "Price rejected from 50-day moving average overhead."},
    {"pattern_name": "MA200 Resistance Rejection",       "type": "strategy", "direction": "bearish",  "description": "Price rejected from 200-day moving average overhead."},
    {"pattern_name": "Positive Momentum Cross",          "type": "strategy", "direction": "bullish",  "description": "10-period ROC crosses above zero. Positive momentum shift."},
    {"pattern_name": "Negative Momentum Cross",          "type": "strategy", "direction": "bearish",  "description": "10-period ROC crosses below zero. Negative momentum shift."},
    {"pattern_name": "NR7 Compression",                  "type": "strategy", "direction": "neutral",  "description": "Narrowest range of last 7 bars. Volatility squeeze before breakout."},
    {"pattern_name": "Gap Up",                           "type": "strategy", "direction": "bullish",  "description": "Today opened >1% above prior close. Bullish momentum gap."},
    {"pattern_name": "Gap Down",                         "type": "strategy", "direction": "bearish",  "description": "Today opened >1% below prior close. Bearish gap momentum."},
    {"pattern_name": "Ichimoku TK Cross Bullish",        "type": "strategy", "direction": "bullish",  "description": "Tenkan crosses above Kijun. Bullish Ichimoku signal."},
    {"pattern_name": "Ichimoku TK Cross Bearish",        "type": "strategy", "direction": "bearish",  "description": "Tenkan crosses below Kijun. Bearish Ichimoku signal."},
    {"pattern_name": "Above Ichimoku Cloud",             "type": "strategy", "direction": "bullish",  "description": "Price above Senkou A and B. Strong bullish environment."},
    {"pattern_name": "Below Ichimoku Cloud",             "type": "strategy", "direction": "bearish",  "description": "Price below Senkou A and B. Strong bearish environment."},
    {"pattern_name": "Ichimoku Cloud Breakout",          "type": "strategy", "direction": "bullish",  "description": "Price just broke above Ichimoku cloud from below."},
    {"pattern_name": "Swing Uptrend Continuation",       "type": "strategy", "direction": "bullish",  "description": "Pattern of higher highs and higher lows confirms uptrend."},
    {"pattern_name": "Swing Downtrend Continuation",     "type": "strategy", "direction": "bearish",  "description": "Pattern of lower highs and lower lows confirms downtrend."},
    {"pattern_name": "Elder Impulse Bullish",            "type": "strategy", "direction": "bullish",  "description": "EMA13 rising + MACD histogram increasing. Both pointing up."},
    {"pattern_name": "Elder Impulse Bearish",            "type": "strategy", "direction": "bearish",  "description": "EMA13 falling + MACD histogram decreasing. Both pointing down."},
    # ── Market analysis patterns ───────────────────────────────────────────
    {"pattern_name": "Dow Primary Uptrend",          "type": "market_analysis", "direction": "bullish", "description": "Dow Theory: higher highs and higher lows across 4+ swings."},
    {"pattern_name": "Dow Primary Downtrend",        "type": "market_analysis", "direction": "bearish", "description": "Dow Theory: lower highs and lower lows across 4+ swings."},
    {"pattern_name": "Dow Trend Reversal Bullish",   "type": "market_analysis", "direction": "bullish", "description": "First higher swing low after a downtrend. Potential trend reversal."},
    {"pattern_name": "Dow Trend Reversal Bearish",   "type": "market_analysis", "direction": "bearish", "description": "First lower swing high after an uptrend. Potential trend reversal."},
    {"pattern_name": "Elliott Wave 3 Upswing",       "type": "market_analysis", "direction": "bullish", "description": "Price in Wave 3 — the strongest and longest wave of the impulse."},
    {"pattern_name": "Elliott Wave 2 Pullback",      "type": "market_analysis", "direction": "bullish", "description": "Wave 2 pullback creates an entry opportunity before Wave 3."},
    {"pattern_name": "Elliott Wave 5 in Progress",   "type": "market_analysis", "direction": "neutral",  "description": "Price completing the 5th wave — end of impulse approaching."},
    {"pattern_name": "Wyckoff Spring",               "type": "market_analysis", "direction": "bullish", "description": "Undercut of support followed by recovery. High-confidence reversal."},
    {"pattern_name": "Wyckoff Sign of Strength",     "type": "market_analysis", "direction": "bullish", "description": "Breakout above resistance with high volume. Markup phase beginning."},
    {"pattern_name": "Wyckoff Sign of Weakness",     "type": "market_analysis", "direction": "bearish", "description": "Breakdown below support with high volume. Markdown phase beginning."},
    {"pattern_name": "Wyckoff Upthrust",             "type": "market_analysis", "direction": "bearish", "description": "False breakout above resistance then reversal. Distribution complete."},
    {"pattern_name": "Wyckoff Selling Climax",       "type": "market_analysis", "direction": "bullish", "description": "Sharp drop with volume spike followed by immediate bounce."},
    {"pattern_name": "52-Week High Breakout",        "type": "market_analysis", "direction": "bullish", "description": "Price breaks to new 52-week high. Strong momentum signal."},
    {"pattern_name": "52-Week Low Reversal",         "type": "market_analysis", "direction": "bullish", "description": "Price bounces from 52-week low. Potential bottom."},
    {"pattern_name": "20-Day High Breakout",         "type": "market_analysis", "direction": "bullish", "description": "Price breaks above 20-day high."},
    {"pattern_name": "20-Day Low Breakdown",         "type": "market_analysis", "direction": "bearish", "description": "Price breaks below 20-day low."},
    {"pattern_name": "Healthy Uptrend",              "type": "market_analysis", "direction": "bullish", "description": "Rising price with rising volume confirms uptrend health."},
    {"pattern_name": "Distribution Warning",         "type": "market_analysis", "direction": "bearish", "description": "Falling price with rising volume — distribution selling."},
    # ── Financial Astrology & Gann Cycles ─────────────────────────────────
    {"pattern_name": "Bradley Siderograph",          "type": "astro", "direction": "neutral", "description": "Donald Bradley's 1947 planetary aspect timing index. Market turns often occur within 4 weeks of a Bradley high or low."},
    {"pattern_name": "Bradley High Turn",            "type": "astro", "direction": "bearish", "description": "Upcoming Bradley Siderograph peak — potential market turning window."},
    {"pattern_name": "Bradley Low Turn",             "type": "astro", "direction": "bullish", "description": "Upcoming Bradley Siderograph trough — potential bullish turning window."},
    {"pattern_name": "Mercury Retrograde",           "type": "astro", "direction": "bearish", "description": "Mercury appears to move backward. Historically associated with communication breakdowns, contract delays, and tech volatility."},
    {"pattern_name": "New Moon",                     "type": "astro", "direction": "bullish", "description": "New Moon phase. Associated with new beginnings and accumulation in financial astrology."},
    {"pattern_name": "Full Moon",                    "type": "astro", "direction": "bearish", "description": "Full Moon phase. Often associated with peak sentiment and increased volatility."},
    {"pattern_name": "Planetary Aspect Cluster",     "type": "astro", "direction": "neutral", "description": "Multiple major planetary aspects forming simultaneously. High-energy turning-point window."},
    {"pattern_name": "Planetary SQ9 Levels",         "type": "astro", "direction": "neutral", "description": "Square of Nine price levels derived from current planetary longitudes. Key support/resistance zones."},
]


def _ew_scan_one(sym: str, resolution_minutes: int = 0) -> list[dict]:
    """Run Elliott Wave detection for one symbol and return scan-format entries.

    Converts the rich EW result into zero or more pattern entries that look
    identical to the entries produced by `service.detect_for_symbol()` so they
    can be merged into the main scan list and rendered by the same UI.

    Scan criteria (one entry per actionable condition):
      - Impulse complete         → expecting ABC correction     (status READY)
      - ABC complete             → new impulse imminent         (status BREAKOUT)
      - W4 retrace in progress   → W5 extension setup           (status FORMING)
      - W2 retrace in progress   → W3 (strongest) setup         (status FORMING)
      - In W3                    → ride the trend               (status FORMING)
    """
    try:
        # Re-use EW cache if available — avoids re-running heavy detection
        cache_key = f"{sym}::{resolution_minutes}:780"
        cached = _ew_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < _EW_CACHE_TTL:
            resp = cached[0]
        else:
            # Build a minimal DataFrame and run detection
            from pattern_engine.service import _fetch_intraday_via_yfinance
            from pattern_engine.elliott_wave import detect_elliott_waves
            from database.connection import get_session
            from database.models import Instrument, PriceData
            from sqlalchemy import select
            import pandas as pd

            df = None
            if resolution_minutes > 0:
                df = _fetch_intraday_via_yfinance(sym, resolution_minutes)
            else:
                with get_session() as session:
                    instrument = session.execute(
                        select(Instrument).where(Instrument.symbol == sym)
                    ).scalar_one_or_none()
                    if instrument:
                        rows = session.execute(
                            select(
                                PriceData.date, PriceData.open, PriceData.high,
                                PriceData.low, PriceData.close, PriceData.volume,
                            )
                            .where(PriceData.instrument_id == instrument.id)
                            .order_by(PriceData.date.desc())
                            .limit(780)
                        ).all()
                        if rows:
                            df = pd.DataFrame(
                                [{"date": r.date, "open": float(r.open), "high": float(r.high),
                                  "low": float(r.low), "close": float(r.close),
                                  "volume": int(r.volume or 0)} for r in reversed(rows)]
                            )
            if df is None or df.empty:
                return []

            result = detect_elliott_waves(sym, df, resolution_minutes=resolution_minutes)
            resp = {"result": result}
            _ew_cache[cache_key] = (resp, time.time())

        result = resp.get("result") or resp  # handle both wrapped and raw

    except Exception:
        return []

    # ── Extract key fields ──────────────────────────────────────────────
    today       = datetime.utcnow().date().isoformat()
    cp          = result.get("current_position") or {}
    best_count  = result.get("best_count")   # serialized dict
    waves_list  = result.get("waves", [])    # WaveCount objects

    if not cp and not best_count:
        return []

    phase       = cp.get("phase", "")
    wave_num    = cp.get("wave_number", 0)
    direction   = cp.get("direction", "up")
    confidence  = cp.get("confidence", 0.0)
    strength    = cp.get("strength", "moderate")
    action      = cp.get("action", "")
    is_up       = direction == "up"
    bull_bear   = "bullish" if is_up else "bearish"

    # Try to derive price levels from best_count waves
    entry_price = target = invalidation = None
    if best_count and best_count.get("waves"):
        bw = best_count["waves"]
        if bw:
            last_wave = bw[-1]
            entry_price = last_wave.get("end_price")
        if len(bw) >= 5:
            w1_size = abs(bw[0].get("end_price", 0) - bw[0].get("start_price", 0))
            w5_end  = bw[4].get("end_price", entry_price or 0)
            impulse_range = abs(w5_end - bw[0].get("start_price", w5_end))
            # Target: next impulse ~= prior impulse from correction low/high
            target        = round(w5_end + (1 if is_up else -1) * impulse_range, 2)
            invalidation  = round(w5_end - (1 if is_up else -1) * impulse_range * 0.382, 2)

    entries: list[dict] = []

    def _entry(name: str, status: str, conf: float, bull: str, tgt=None, inv=None, lvl=None) -> dict:
        return {
            "symbol":             sym,
            "date":               today,
            "tf":                 "daily" if resolution_minutes == 0 else f"{resolution_minutes}m",
            "pattern_name":       name,
            "pattern_category":   "elliott",
            "direction":          bull,
            "status":             status,
            "confidence":         round(float(conf), 2),
            "breakout_level":     lvl or entry_price,
            "target":             tgt or target,
            "invalidation_level": inv or invalidation,
            "phase":              phase,
            "phase_label":        action,
            "events":             [],
        }

    # ── Classify and emit entries ───────────────────────────────────────
    if phase == "impulse":
        if wave_num == 5:
            # W5 in progress → approaching impulse completion → watch for reversal
            entries.append(_entry(
                "EW Wave 5 — Final Push", "FORMING",
                min(confidence, 72), bull_bear,
            ))
        elif wave_num == 4:
            # W4 retrace → W5 extension setup
            entries.append(_entry(
                "EW Wave 4 — Retrace Setup", "FORMING",
                min(confidence, 68), bull_bear,
            ))
        elif wave_num == 3:
            # In W3 — strongest wave, trend continuation
            entries.append(_entry(
                "EW Wave 3 — Trend Acceleration", "FORMING",
                min(confidence, 75), bull_bear,
            ))
        elif wave_num == 2:
            # W2 retrace → best R/R entry for W3
            entries.append(_entry(
                "EW Wave 2 — Entry Setup", "READY",
                min(confidence, 78), bull_bear,
            ))

    elif phase == "correction":
        sub = cp.get("sub_phase", "")
        wave_lbl = cp.get("wave_in_progress", "")
        if "complete" in action.lower() or "C" in wave_lbl:
            # ABC complete → new impulse imminent
            entries.append(_entry(
                "EW Correction Complete — New Impulse", "BREAKOUT",
                min(confidence, 82), bull_bear,
            ))
        elif "B" in wave_lbl:
            # In wave B → C leg (final correction leg) coming
            entries.append(_entry(
                "EW Wave B — C Leg Expected", "FORMING",
                min(confidence, 65), "bearish" if is_up else "bullish",
            ))
        else:
            # Correction in progress (A leg or early)
            entries.append(_entry(
                "EW Impulse Complete — ABC Correction", "READY",
                min(confidence, 70), "bearish" if is_up else "bullish",
            ))

    return entries


@router.get("/scan")
def scan_all_patterns(
    invalidate: bool = False,
    tf: str = Query(default="", description="Timeframe: 1m 5m 15m 30m 1h 2h 4h (omit for daily)"),
):
    """Run pattern detection for all tracked instruments in parallel.

    Returns a flat list of active patterns (FORMING / READY / BREAKOUT / COMPLETED)
    across every instrument, sorted by confidence descending.

    Pass ?tf=4h to run intraday detection at that resolution (separate cache slot).
    Results are cached 5 minutes per timeframe.  Pass ?invalidate=true to force refresh.
    """
    global _scan_cache
    resolution_minutes = _TF_RESOLUTION.get(tf.lower().strip(), 0)
    cache_key = tf.lower().strip() if resolution_minutes > 0 else "daily"
    now = time.time()

    cached_results, cached_ts = _scan_cache.get(cache_key, (None, 0.0))
    if not invalidate and cached_results is not None and (now - cached_ts) < _SCAN_TTL:
        return cached_results

    from database.connection import get_session
    from database.models import Instrument
    from sqlalchemy import select

    with get_session() as session:
        symbols: list[str] = [
            row.symbol
            for row in session.execute(select(Instrument)).scalars().all()
        ]

    today = datetime.utcnow().date().isoformat()
    active_statuses = {"FORMING", "READY", "BREAKOUT", "COMPLETED"}

    def _detect_one(sym: str) -> list[dict]:
        try:
            raw = service.detect_for_symbol(sym, resolution_minutes)
        except Exception:
            return []
        out = []
        for p in raw:
            status = p.get("status", "NOT_PRESENT")
            if status not in active_statuses:
                continue
            direction = p.get("direction", "neutral")
            if not direction or direction not in {"bullish", "bearish", "neutral"}:
                direction = "bullish" if p.get("pattern_category") == "chart" else "neutral"
            out.append({
                "symbol":            sym,
                "date":              today,
                "tf":                cache_key,
                "pattern_name":      p.get("pattern_name"),
                "pattern_category":  p.get("pattern_category"),
                "direction":         direction,
                "status":            status,
                "confidence":        round(float(p.get("confidence", 0.0) or 0.0), 2),
                "breakout_level":    p.get("breakout_level"),
                "target":            p.get("target", p.get("projected_target")),
                "invalidation_level": p.get("invalidation_level"),
                "phase":             p.get("phase"),
                "phase_label":       p.get("phase_label"),
                "events":            p.get("events", []),
            })
        return out

    results: list[dict] = []
    # Run traditional pattern detection + EW detection in parallel pools
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_detect_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            results.extend(future.result())

    # EW scan runs daily only (EW detection is heavy; intraday EW is noisy)
    if resolution_minutes == 0:
        with ThreadPoolExecutor(max_workers=4) as ew_pool:
            ew_futures = {ew_pool.submit(_ew_scan_one, sym, 0): sym for sym in symbols}
            for future in as_completed(ew_futures):
                results.extend(future.result())

    results.sort(key=lambda x: x.get("confidence", 0) or 0, reverse=True)

    _scan_cache[cache_key] = (results, now)
    return results


@router.get("/catalogue")
def get_pattern_catalogue():
    """Return patterns currently supported by the refactored engine."""
    return SUPPORTED_PATTERNS


@router.get("/{symbol}/stage4")
def get_stage4_signal(symbol: str):
    """Stage 4 — Pattern Signal for the 5-stage trading decision pipeline.

    Aggregates all active chart/harmonic/wyckoff/gann patterns for a symbol
    into a single Stage 4 signal object:

      STAGE 1 REGIME      → environment filter
      STAGE 2 CORE SIGNALS→ volume / momentum / structure
      STAGE 3 AI PROB     → ML model probability
    ▶ STAGE 4 PATTERN     → this endpoint
      STAGE 5 PRICE LEVELS→ entry / stop / target (derived from pattern levels)

    Returns:
      signal       : "bullish" | "bearish" | "neutral"
      score        : 0-100 weighted pattern conviction
      active       : list of active patterns (FORMING/READY/BREAKOUT)
      top_pattern  : highest-confidence active pattern
      entry        : best breakout_level from top pattern
      target       : best target from top pattern
      stop         : best invalidation_level from top pattern
      pattern_count: number of active patterns
      bull_count   : bullish active patterns
      bear_count   : bearish active patterns
    """
    sym = symbol.upper()
    # Use cached daily scan if available; otherwise run live
    cached_results, _ = _scan_cache.get("daily", (None, 0.0))
    if cached_results is not None:
        active = [
            p for p in cached_results
            if p.get("symbol") == sym
            and p.get("status") in {"FORMING", "READY", "BREAKOUT", "COMPLETED"}
        ]
    else:
        # Live detection (slower)
        try:
            raw = service.detect_for_symbol(sym, 0)
        except Exception:
            raw = []
        today = datetime.utcnow().date().isoformat()
        active = []
        for p in raw:
            if p.get("status") not in {"FORMING", "READY", "BREAKOUT", "COMPLETED"}:
                continue
            direction = p.get("direction", "neutral")
            active.append({
                "symbol": sym, "date": today, "tf": "daily",
                "pattern_name": p.get("pattern_name"),
                "pattern_category": p.get("pattern_category"),
                "direction": direction,
                "status": p.get("status"),
                "confidence": round(float(p.get("confidence", 0.0) or 0.0), 2),
                "breakout_level": p.get("breakout_level"),
                "target": p.get("target", p.get("projected_target")),
                "invalidation_level": p.get("invalidation_level"),
            })

    if not active:
        return {
            "symbol": sym, "signal": "neutral", "score": 0,
            "active": [], "top_pattern": None,
            "entry": None, "target": None, "stop": None,
            "pattern_count": 0, "bull_count": 0, "bear_count": 0,
        }

    # Weight by confidence
    bull_score = sum(
        p["confidence"] for p in active if p.get("direction") == "bullish"
    )
    bear_score = sum(
        p["confidence"] for p in active if p.get("direction") == "bearish"
    )
    bull_count = sum(1 for p in active if p.get("direction") == "bullish")
    bear_count = sum(1 for p in active if p.get("direction") == "bearish")

    # Weighted net score: +100 = all bullish, -100 = all bearish
    total_score = bull_score + bear_score
    if total_score > 0:
        net = (bull_score - bear_score) / total_score * 100
    else:
        net = 0.0

    if net > 15:
        signal = "bullish"
    elif net < -15:
        signal = "bearish"
    else:
        signal = "neutral"

    # Top pattern = highest confidence active
    top = max(active, key=lambda x: x.get("confidence", 0) or 0)

    return {
        "symbol":        sym,
        "signal":        signal,
        "score":         round(abs(net), 1),
        "net_score":     round(net, 1),
        "active":        sorted(active, key=lambda x: x.get("confidence", 0) or 0, reverse=True),
        "top_pattern":   top.get("pattern_name"),
        "entry":         top.get("breakout_level"),
        "target":        top.get("target"),
        "stop":          top.get("invalidation_level"),
        "pattern_count": len(active),
        "bull_count":    bull_count,
        "bear_count":    bear_count,
    }


@router.get("/{symbol}/mtf")
def get_mtf_pattern_summary(symbol: str):
    """
    Return pattern count per timeframe for a specific symbol.

    Reads from the in-process scan cache — zero extra computation.
    Returns cached=false for any TF whose scan has not run yet.
    """
    active_statuses = {"FORMING", "READY", "BREAKOUT", "COMPLETED"}
    sym = symbol.upper()
    tf_keys = [
        ("daily", "Daily"),
        ("4h",    "4h"),
        ("1h",    "1h"),
        ("30m",   "30m"),
        ("15m",   "15m"),
    ]
    result: dict[str, dict] = {}
    for tf_key, tf_label in tf_keys:
        cached_results, _ = _scan_cache.get(tf_key, (None, 0.0))
        if cached_results is not None:
            sym_active = [
                p for p in cached_results
                if p.get("symbol") == sym and p.get("status") in active_statuses
            ]
            top_conf = max(
                (p.get("confidence", 0) or 0 for p in sym_active),
                default=0,
            )
            result[tf_key] = {
                "label":          tf_label,
                "count":          len(sym_active),
                "top_confidence": round(float(top_conf), 1),
                "cached":         True,
            }
        else:
            result[tf_key] = {
                "label":          tf_label,
                "count":          0,
                "top_confidence": 0,
                "cached":         False,
            }
    return {"symbol": sym, "tfs": result}


@router.get("/{symbol}/chart-overlay")
def get_chart_overlay(
    symbol: str,
    bars: int = Query(default=780, ge=30, le=3000),
    tf: str = Query(default="", description="Timeframe: 1m 5m 15m 30m 1h (omit for daily)"),
):
    """Return OHLCV bars AND detected patterns from the SAME DataFrame.

    This guarantees perfect pattern-candle alignment because both the
    chart data and the pattern overlay points come from identical bars.
    Pattern points use INTEGER indices into the returned bars array.
    """
    from fastapi import HTTPException

    resolution_minutes = _TF_RESOLUTION.get(tf.lower().strip(), 0)
    sym = symbol.upper()

    # ── Load bars using the SAME logic as PatternService ─────────────────
    df = None
    if resolution_minutes > 0:
        from pattern_engine.service import _fetch_intraday_via_yfinance
        df = _fetch_intraday_via_yfinance(sym, resolution_minutes)
    else:
        from database.connection import get_session
        from database.models import Instrument, PriceData
        from sqlalchemy import select
        with get_session() as session:
            instrument = session.execute(
                select(Instrument).where(Instrument.symbol == sym)
            ).scalar_one_or_none()
            if instrument:
                rows = session.execute(
                    select(
                        PriceData.date, PriceData.open, PriceData.high,
                        PriceData.low, PriceData.close, PriceData.volume,
                    )
                    .where(PriceData.instrument_id == instrument.id)
                    .order_by(PriceData.date.desc())
                    .limit(bars + 50)
                ).all()
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                    df["date"] = df["date"].astype(str)
                    df = df.sort_values("date").reset_index(drop=True)

        if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 20:
            from pattern_engine.service import _fetch_daily_via_yfinance
            df = _fetch_daily_via_yfinance(sym, limit=bars + 50)

    if df is None or len(df) < 20:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = df.tail(bars).reset_index(drop=True)

    # ── Run detection on these exact bars ─────────────────────────────────
    from pattern_engine.chart_patterns import detect_all as detect_all_chart
    raw_patterns = detect_all_chart(sym, df, resolution_minutes)
    try:
        from pattern_engine.harmonic_patterns import detect as detect_harmonic
        raw_patterns.extend(detect_harmonic(sym, df))
    except Exception:
        pass
    try:
        from pattern_engine.wyckoff_structures import detect as detect_wyckoff
        wyckoff_result = detect_wyckoff(sym, df)
        if isinstance(wyckoff_result, list):
            raw_patterns.extend(wyckoff_result)
        elif isinstance(wyckoff_result, dict):
            raw_patterns.append(wyckoff_result)
    except Exception:
        pass
    try:
        from pattern_engine.measured_move import detect as detect_mm
        raw_patterns.extend(detect_mm(sym, df))
    except Exception:
        pass

    ACTIVE = {"FORMING", "READY", "BREAKOUT", "COMPLETED"}
    last_idx = len(df) - 1
    active = [
        p for p in raw_patterns
        if p.get("status") in ACTIVE
        and p.get("points")
        and len(p.get("points", [])) >= 2
    ]

    # ── Relevance scoring: prefer patterns involving current candles ──
    # A pattern whose overlay lines reach to/past last_idx is far more
    # useful than one that only connects historical bars.
    def _min_max_index(p: dict) -> tuple[int, int]:
        """Get the leftmost and rightmost bar index from points + overlay_lines."""
        mn, mx = last_idx, 0
        for pt in p.get("points", []):
            if isinstance(pt, (list, tuple)) and len(pt) >= 1:
                v = int(pt[0]) if isinstance(pt[0], (int, float)) else 0
                mn = min(mn, v)
                mx = max(mx, v)
        for line in p.get("overlay_lines", []):
            if isinstance(line, (list, tuple)):
                for pt in line:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 1:
                        v = int(pt[0]) if isinstance(pt[0], (int, float)) else 0
                        mn = min(mn, v)
                        mx = max(mx, v)
        return mn, mx

    def _relevance_score(p: dict) -> float:
        """Combined score: 40% recency + 30% span + 30% confidence.

        Patterns spanning the entire chart AND reaching current candles
        score highest.  A full-chart descending channel (200 bars, reaching
        current price) beats a 20-bar triangle with higher confidence.
        """
        mn, mx = _min_max_index(p)
        recency = min(mx / max(last_idx, 1), 1.0)
        span = (mx - mn) / max(last_idx, 1)  # 0..1: fraction of chart covered
        conf = float(p.get("confidence", p.get("probability", 0)) or 0) / 100.0
        return recency * 0.40 + span * 0.30 + conf * 0.30

    active.sort(key=_relevance_score, reverse=True)
    active = active[:5]

    # ── Serialize bars ────────────────────────────────────────────────────
    bar_list = []
    for i, row in df.iterrows():
        d = str(row["date"])
        try:
            from datetime import datetime as _dt
            if "T" in d or " " in d:
                ts_ms = int(_dt.fromisoformat(d.replace("Z", "+00:00")).timestamp() * 1000)
            else:
                ts_ms = int(_dt.strptime(d, "%Y-%m-%d").timestamp() * 1000)
        except Exception:
            ts_ms = 0
        bar_list.append({
            "timestamp": ts_ms,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row.get("volume", 0) or 0),
        })

    # ── Serialize patterns with RAW integer indices ──────────────────────
    pattern_list = []
    for p in active:
        pattern_list.append({
            "pattern_name": p.get("pattern_name"),
            "pattern_category": p.get("pattern_category", "chart"),
            "status": p.get("status"),
            "direction": p.get("direction", "neutral"),
            "confidence": round(float(p.get("confidence", p.get("probability", 0)) or 0), 2),
            "breakout_level": p.get("breakout_level"),
            "projected_target": p.get("target", p.get("projected_target")),
            "invalidation_level": p.get("invalidation_level"),
            "points": p.get("points", []),
            "point_labels": p.get("point_labels"),
            "overlay_lines": p.get("overlay_lines", []),
            "overlay_line_roles": p.get("overlay_line_roles", []),
            "fill_zone": p.get("fill_zone"),
            "event_points": p.get("event_points", []),
            "prz_low": p.get("prz_low"),
            "prz_high": p.get("prz_high"),
        })

    return {
        "symbol": sym,
        "bars": bar_list,
        "patterns": pattern_list,
    }


@router.get("/{symbol}")
def get_patterns_for_symbol(
    symbol: str,
    limit: int = 500,
    tf: str = Query(default="", description="Timeframe: 1m 5m 15m 30m 1h (omit for daily)"),
):
    """Run pattern detection for a symbol.

    Pass ?tf=30m for intraday bars (fetched via yfinance).
    Omit tf (or use daily/1d) to use daily bars from the DB with yfinance fallback.
    """
    resolution_minutes = _TF_RESOLUTION.get(tf.lower().strip(), 0)
    raw_patterns = service.detect_for_symbol(symbol.upper(), resolution_minutes)

    # ── Output filtering: quality over quantity ────────────────────────────
    # 1. Remove non-active statuses (FAILED, NOT_PRESENT are noise)
    ACTIVE = {"FORMING", "READY", "BREAKOUT", "COMPLETED"}
    raw_patterns = [p for p in raw_patterns if p.get("status") in ACTIVE]

    # 2. Per-category caps — prevent any single category from flooding
    CATEGORY_CAPS = {
        "harmonic":        2,   # max 2 harmonic patterns (e.g. Gartley + Crab)
        "chart":           4,   # max 4 chart patterns
        "gann":            1,   # only 1 Gann overlay
        "wyckoff":         1,   # only 1 Wyckoff structure
        "candlestick":     2,   # max 2 candlestick patterns (only fresh ones survive)
        "behavioral":      2,
        "indicator":       2,
        "volume":          2,
        "measured_move":   1,
        "strategy":        3,   # limit strategy signal spam
        "market_analysis": 2,
        "astro":           1,   # 1 astro pattern max — informational only
    }
    cat_count: dict[str, int] = {}
    filtered: list[dict] = []
    for p in sorted(raw_patterns, key=lambda x: x.get("confidence", 0), reverse=True):
        cat = p.get("pattern_category", "chart")
        cap = CATEGORY_CAPS.get(cat, 2)
        if cat_count.get(cat, 0) < cap:
            cat_count[cat] = cat_count.get(cat, 0) + 1
            filtered.append(p)

    # 3. Minimum confidence floor after category filtering (belt + suspenders)
    filtered = [p for p in filtered if (p.get("confidence") or 0) >= 55.0]

    # 4. Resolve contradictory pattern pairs — keep only the higher-confidence one
    # Each tuple: (pattern_A, pattern_B) — only one should survive per group
    CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
        ("Gap Up",                           "Gap Down"),
        ("RSI(2) Oversold Setup",            "RSI(2) Overbought Exit"),
        ("Stochastic Pop",                   "Stochastic Drop"),
        ("CCI Correction Bullish",           "CCI Correction Bearish"),
        ("Golden Cross Setup",               "Death Cross Setup"),
        ("MACD Bullish Crossover",           "MACD Bearish Crossover"),
        ("MACD Zero Line Cross Up",          "MACD Zero Line Cross Down"),
        ("RSI Regular Bullish Divergence",   "RSI Regular Bearish Divergence"),
        ("Dow Trend Reversal Signal Bullish", "Dow Trend Reversal Signal Bearish"),
        ("Ichimoku TK Cross Bullish",        "Ichimoku TK Cross Bearish"),
        ("Above Ichimoku Cloud",             "Below Ichimoku Cloud"),
        ("Elder Impulse Bullish",            "Elder Impulse Bearish"),
        ("Healthy Uptrend",                  "Healthy Downtrend"),
    ]
    present_names = {p.get("pattern_name", ""): p for p in filtered}
    to_remove: set[str] = set()
    for name_a, name_b in CONTRADICTORY_PAIRS:
        if name_a in present_names and name_b in present_names:
            pa, pb = present_names[name_a], present_names[name_b]
            # Remove the lower-confidence one; prefer BREAKOUT/READY over FORMING
            def _priority(pat: dict) -> float:
                status_bonus = {"BREAKOUT": 10, "READY": 5, "COMPLETED": 8, "FORMING": 0}
                return (pat.get("confidence") or 0) + status_bonus.get(pat.get("status", ""), 0)
            if _priority(pa) >= _priority(pb):
                to_remove.add(name_b)
            else:
                to_remove.add(name_a)
    filtered = [p for p in filtered if p.get("pattern_name", "") not in to_remove]

    # 5. Global cap — user wants quality not quantity
    patterns = filtered[:min(limit, 12)]

    now = datetime.utcnow().date().isoformat()
    out = []
    for p in patterns:
        breakout = p.get("breakout_level")
        target = p.get("target", p.get("projected_target"))
        status = p.get("status", "NOT_PRESENT")
        # Use detector-provided direction if available, else infer
        direction = p.get("direction", "neutral")
        if not direction or direction not in {"bullish", "bearish", "neutral"}:
            direction = "bullish" if p.get("pattern_category") == "chart" else "neutral"
        message = None
        if status == "NOT_PRESENT":
            message = "Pattern not detected"
        elif status in {"FORMING", "READY"}:
            message = "Pattern forming"
        out.append(
            {
                "date": now,
                "pattern_name": p.get("pattern_name"),
                "pattern_category": p.get("pattern_category"),
                "pattern_type": "chart",
                "direction": direction,
                "strength": round((p.get("confidence", 0.0) or 0.0) / 100.0, 4),
                "price_level": breakout,
                "status": status,
                "breakout_level": breakout,
                "invalidation_level": p.get("invalidation_level"),
                "projected_target": target,
                "target": target,
                "probability": p.get("confidence", 0.0),
                "confidence": p.get("confidence", 0.0),
                "points": p.get("points", []),
                "point_labels": p.get("point_labels"),
                "overlay_lines": p.get("overlay_lines", []),
                "message": message,
                # Wyckoff enriched fields
                "phase": p.get("phase"),
                "phase_label": p.get("phase_label"),
                "events": p.get("events", []),
                "support_level": p.get("support_level"),
                "resistance_level": p.get("resistance_level"),
                # Wyckoff/chart event point coordinates for chart annotation
                "event_points": p.get("event_points", []),
                # Per-segment styling roles (Chair / Cup & Handle)
                "overlay_line_roles": p.get("overlay_line_roles", []),
                # Gann enriched fields
                "fan_lines": p.get("fan_lines", []),
                "retracement_levels": p.get("retracement_levels", []),
                "time_cycles": p.get("time_cycles", []),
                # Cup & Handle named key prices
                "cup_left_price": p.get("cup_left_price"),
                "cup_right_price": p.get("cup_right_price"),
                "cup_bottom_price": p.get("cup_bottom_price"),
                # Harmonic-specific
                "ratios": p.get("ratios"),
                "prz_low": p.get("prz_low"),
                "prz_high": p.get("prz_high"),
                # Shaded fill zone between two trendlines (channels, wedges, triangles)
                "fill_zone": p.get("fill_zone"),
                # Astrology / financial cycles fields
                "bradley_series": p.get("bradley_series"),
                "bradley_turning_points": p.get("bradley_turning_points"),
                "upcoming_turning_points": p.get("upcoming_turning_points"),
                "turn_date": p.get("turn_date"),
                "turn_type": p.get("turn_type"),
                "days_until": p.get("days_until"),
                "retro_start": p.get("retro_start"),
                "retro_end": p.get("retro_end"),
                "retro_ongoing": p.get("retro_ongoing"),
                "phase_date": p.get("phase_date"),
                "days_delta": p.get("days_delta"),
                "ingress_date": p.get("ingress_date"),
                "from_sign": p.get("from_sign"),
                "to_sign": p.get("to_sign"),
                "aspects": p.get("aspects"),
                "bullish_aspects": p.get("bullish_aspects"),
                "bearish_aspects": p.get("bearish_aspects"),
                "aspect_date": p.get("aspect_date"),
                "sq9_planetary_levels": p.get("sq9_planetary_levels"),
                # Sub-chart data series (large — only present on specific astro patterns)
                "raw_bradley_series": p.get("raw_bradley_series"),
                "planet_series": p.get("planet_series"),
            }
        )
    return out


@router.get("/{symbol}/elliott-waves")
def get_elliott_waves(
    symbol: str,
    bars: int = Query(default=780, ge=50, le=3000),
    tf: str = Query(default="", description="Timeframe: 1m 5m 15m 30m 1h 1w 1mo (omit for daily)"),
    nocache: bool = Query(default=False, description="Skip cache and force fresh detection"),
):
    """Multi-degree Elliott Wave analysis with pivot detection and Fibonacci scoring.

    Returns:
    - Wave counts at primary, intermediate, and minor degrees
    - Current wave position assessment with trade bias
    - Serialized wave data with bar indices for chart rendering
    """
    from fastapi import HTTPException
    import logging as _logging
    _log = _logging.getLogger("roboalgo.ew")

    tf_lower = tf.lower().strip()
    sym = symbol.upper()

    # ── Cache check — return instantly if we have a recent result ────────
    cache_key = f"{sym}:{tf_lower}:{bars}"
    if not nocache:
        cached = _ew_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < _EW_CACHE_TTL:
            _log.debug("EW cache hit for %s", cache_key)
            return cached[0]

    _t0 = time.time()
    resolution_minutes = _TF_RESOLUTION.get(tf_lower, 0)
    is_weekly = tf_lower in ("1w", "weekly")
    is_monthly = tf_lower in ("1mo", "monthly")

    # Load bars — same logic as chart-overlay for consistency
    df = None
    if resolution_minutes > 0:
        from pattern_engine.service import _fetch_intraday_via_yfinance
        df = _fetch_intraday_via_yfinance(sym, resolution_minutes)
    else:
        # For weekly/monthly we need MORE daily bars to resample from
        fetch_limit = bars + 50
        if is_weekly:
            fetch_limit = max(bars * 7, 3000)  # ~bars weeks of daily data
        elif is_monthly:
            fetch_limit = max(bars * 22, 5000)

        from database.connection import get_session
        from database.models import Instrument, PriceData
        from sqlalchemy import select
        with get_session() as session:
            instrument = session.execute(
                select(Instrument).where(Instrument.symbol == sym)
            ).scalar_one_or_none()
            if instrument:
                rows = session.execute(
                    select(
                        PriceData.date, PriceData.open, PriceData.high,
                        PriceData.low, PriceData.close, PriceData.volume,
                    )
                    .where(PriceData.instrument_id == instrument.id)
                    .order_by(PriceData.date.desc())
                    .limit(fetch_limit)
                ).all()
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                    df["date"] = df["date"].astype(str)
                    df = df.sort_values("date").reset_index(drop=True)

        if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 50:
            try:
                from api.routers.prices import _fetch_any
                prices_df, _ = _fetch_any(sym, limit=fetch_limit)
                if not prices_df.empty and len(prices_df) >= 50:
                    prices_df = prices_df.reset_index()
                    prices_df.columns = [c.lower() for c in prices_df.columns]
                    df = prices_df
            except Exception:
                pass

        if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 50:
            from pattern_engine.service import _fetch_daily_via_yfinance
            df = _fetch_daily_via_yfinance(sym, limit=fetch_limit)

    if df is None or len(df) < 50:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")

    # Resample to weekly or monthly if requested
    if is_weekly or is_monthly:
        import pandas as pd
        df["date"] = pd.to_datetime(df["date"])
        freq = "W" if is_weekly else "ME"
        resampled = df.set_index("date").resample(freq).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["open", "close"]).reset_index()
        resampled["date"] = resampled["date"].astype(str)
        df = resampled
        # For weekly/monthly, use a different resolution for EW pivot sizing
        resolution_minutes = 10080 if is_weekly else 43200  # week=10080min, month=43200min

    df = df.tail(bars).reset_index(drop=True)
    # Drop rows with NaN in OHLC (critical for EW detection)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df.get("volume", 0).fillna(0)

    # Run Elliott Wave detection
    from pattern_engine.elliott_wave import detect_elliott_waves, serialize_all_waves
    result = detect_elliott_waves(sym, df, resolution_minutes=resolution_minutes)

    # Serialize bars for chart alignment
    bar_list = []
    for i, row in df.iterrows():
        d = str(row["date"])
        try:
            from datetime import datetime as _dt
            if "T" in d or " " in d:
                ts_ms = int(_dt.fromisoformat(d.replace("Z", "+00:00")).timestamp() * 1000)
            else:
                ts_ms = int(_dt.strptime(d, "%Y-%m-%d").timestamp() * 1000)
        except Exception:
            ts_ms = 0
        bar_list.append({
            "timestamp": ts_ms,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row.get("volume", 0) or 0),
        })

    # Find CONNECTED impulse → correction pair (the full EW cycle)
    from pattern_engine.elliott_wave import WaveType, WaveDirection, _serialize_wave_count as _ser, compute_ew_channels
    all_waves = result["waves"]
    total_bars = len(bar_list)

    best_impulse = None
    linked_correction = None

    impulse_all = [wc for wc in all_waves if wc.wave_type == WaveType.IMPULSE]

    if is_weekly or is_monthly:
        # ── MACRO SELECTION (weekly/monthly) ──
        # For macro charts, prefer the WIDEST-SPANNING impulse that ALSO
        # terminates near the structural high (ATH for bullish, ATL bearish).
        # An impulse whose Wave 5 ends at the actual chart high is much more
        # useful than one that stops short — that's where price actually went.
        # Score = span_pct * 0.40 + wave_count * 0.20 + conf * 0.15 + extreme_proximity * 0.25
        # Anchor points: the actual high/low of the visible chart range
        global_high = max((b.get("high", 0) or 0) for b in bar_list) if bar_list else 0
        global_low  = min((b.get("low", 1e9) or 1e9) for b in bar_list) if bar_list else 1e9
        price_range = max(global_high - global_low, 1e-9)

        def _macro_score(wc):
            span = wc.waves[-1].end.idx - wc.waves[0].start.idx
            span_pct = span / max(total_bars, 1)
            nw_norm = len(wc.waves) / 5.0
            conf_norm = wc.confidence / 100.0

            # Anchor proximity: how well does this count span between global
            # extremes? Bullish impulse should START near global low and END
            # near global high. Use price-range normalization, not absolute.
            start_price = wc.waves[0].start.price
            end_price = wc.waves[-1].end.price
            if wc.direction == WaveDirection.UP:
                start_anchor = max(0.0, 1.0 - abs(start_price - global_low) / price_range)
                end_anchor   = max(0.0, 1.0 - abs(end_price - global_high) / price_range)
            else:
                start_anchor = max(0.0, 1.0 - abs(start_price - global_high) / price_range)
                end_anchor   = max(0.0, 1.0 - abs(end_price - global_low) / price_range)
            # Cube end_anchor — exact ATH/ATL termination wins decisively over
            # "close to extreme." A wave 5 ending precisely at the chart high
            # is the structurally meaningful count, not one stopping 5% short.
            anchor_score = (start_anchor + end_anchor ** 3) / 2.0

            # Hard bonus: structure terminating within 1% of global extreme
            # is the canonical count the user can SEE on the chart. This
            # tiebreaker is decisive when two counts have similar span/conf.
            if wc.direction == WaveDirection.UP:
                hits_extreme = abs(end_price - global_high) / max(global_high, 1e-9) < 0.01
            else:
                hits_extreme = abs(end_price - global_low) / max(global_low, 1e-9) < 0.01
            extreme_bonus = 0.15 if hits_extreme else 0.0

            # W3 prominence: prefer counts where W3 reaches a significant
            # price level. In a strong bull market, W3 should be at a major
            # structural peak (e.g. BTC $108K before the $126K ATH), not a
            # minor intermediate high. For DOWN impulse, lower W3 = more prominent.
            w3_price = wc.waves[2].end.price if len(wc.waves) > 2 else end_price
            if wc.direction == WaveDirection.UP:
                w3_prominence = (w3_price - global_low) / max(price_range, 1e-9)
            else:
                w3_prominence = (global_high - w3_price) / max(price_range, 1e-9)
            w3_prominence = max(0.0, min(1.0, w3_prominence))

            return (span_pct * 0.18 + nw_norm * 0.08 + conf_norm * 0.09
                    + anchor_score * 0.55 + w3_prominence * 0.10 + extreme_bonus)

        if impulse_all:
            impulse_all.sort(key=_macro_score, reverse=True)
            best_impulse = impulse_all[0]
    else:
        # ── MICRO SELECTION (intraday/daily) ──
        # Prefer recency: partial impulse at the current bar is most useful.
        # Priority: recent partial > recent complete > older partial > any

        # Tier 1: partial impulse at the current bar
        recent_5pct = max(total_bars - int(total_bars * 0.05), total_bars - 20)
        for wc in impulse_all:
            if not wc.is_complete and wc.waves[-1].end.idx >= recent_5pct:
                if best_impulse is None or len(wc.waves) > len(best_impulse.waves) or \
                   (len(wc.waves) == len(best_impulse.waves) and wc.confidence > best_impulse.confidence):
                    best_impulse = wc

        # Tier 2: complete impulse ending very recently
        if best_impulse is None:
            recent_10pct = max(total_bars - int(total_bars * 0.10), total_bars - 50)
            for wc in impulse_all:
                if wc.is_complete and wc.waves[-1].end.idx >= recent_10pct:
                    if best_impulse is None or wc.confidence > best_impulse.confidence:
                        best_impulse = wc

        # Tier 3: partial impulse ending somewhat recently
        if best_impulse is None:
            recent_15pct = max(total_bars - int(total_bars * 0.15), total_bars - 80)
            for wc in impulse_all:
                if not wc.is_complete and wc.waves[-1].end.idx >= recent_15pct:
                    if best_impulse is None or len(wc.waves) > len(best_impulse.waves):
                        best_impulse = wc

        # Tier 4: complete impulse in recent half
        if best_impulse is None:
            recent_half = max(int(total_bars * 0.50), total_bars - 200)
            candidates = [wc for wc in impulse_all if wc.is_complete and wc.waves[-1].end.idx >= recent_half]
            if candidates:
                candidates.sort(key=lambda wc: wc.waves[-1].end.idx * 0.6 + wc.confidence * 0.4, reverse=True)
                best_impulse = candidates[0]

        # Tier 5: any impulse (fallback)
        if best_impulse is None and impulse_all:
            impulse_all.sort(key=lambda wc: wc.confidence, reverse=True)
            best_impulse = impulse_all[0]

    # Find linked correction starting near impulse end
    if best_impulse:
        imp_end_idx = best_impulse.waves[-1].end.idx
        proximity = max(5, int(total_bars * 0.03))  # 3% of bars or at least 5
        for cwc in all_waves:
            if cwc.wave_type in (WaveType.CORRECTIVE, WaveType.TRIANGLE, WaveType.DOUBLE_CORRECTION):
                corr_start_idx = cwc.waves[0].start.idx
                if abs(corr_start_idx - imp_end_idx) <= proximity:
                    linked_correction = cwc
                    break

    import math
    def _sanitize(obj):
        """Replace NaN/Inf with None for JSON serialization."""
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    # Compute EW channels for the best impulse + linked correction
    channels = compute_ew_channels(best_impulse, linked_correction)

    response = _sanitize({
        "symbol": sym,
        "bars": bar_list,
        "wave_counts": serialize_all_waves(result["waves"]),
        "current_position": result["current_position"],
        "best_count": result["best_count"],
        "best_impulse": _ser(best_impulse) if best_impulse else None,
        "linked_correction": _ser(linked_correction) if linked_correction else None,
        "channels": channels,
    })

    # Cache the result and log timing
    _ew_cache[cache_key] = (response, time.time())
    _log.info("EW %s computed in %.1fs (%d counts, %d bars)",
              cache_key, time.time() - _t0,
              len(response.get("wave_counts", [])), len(bar_list))
    return response
