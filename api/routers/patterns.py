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
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_detect_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
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
    patterns = service.detect_for_symbol(symbol.upper(), resolution_minutes)[:limit]
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
