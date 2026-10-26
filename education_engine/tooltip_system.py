"""
education_engine/tooltip_system.py
────────────────────────────────────────────────────────────────────────────────
Tooltip definitions for every key concept in RoboAlgo.

Each tooltip has three layers:
  simple        → 1-sentence plain English (beginner)
  practical     → what it means for THIS market right now
  implication   → what to DO as a trader

Served via API → React renders as hover cards.
"""
from __future__ import annotations

from typing import TypedDict


class TooltipDef(TypedDict):
    term:        str
    category:    str       # gex | signal | pattern | indicator | regime
    simple:      str       # ≤ 15 words
    practical:   str       # ≤ 30 words
    implication: str       # ≤ 30 words
    beginner_emoji: str


TOOLTIPS: dict[str, TooltipDef] = {

    # ── Options / Dealer Positioning ─────────────────────────────────────────

    "gex": {
        "term": "GEX — Gamma Exposure",
        "category": "gex",
        "simple": "Shows how much options dealers will be forced to buy or sell as price moves.",
        "practical": "Negative GEX means dealers amplify moves. Positive GEX means they dampen moves and pin price.",
        "implication": "In negative GEX: expect big fast moves and ride them. In positive GEX: fade extremes.",
        "beginner_emoji": "⚡",
    },
    "gamma_flip": {
        "term": "Gamma Flip Level",
        "category": "gex",
        "simple": "The exact price where market behavior switches from stable to unstable.",
        "practical": "Above gamma flip = calm, pinned market. Below gamma flip = wild, fast moves.",
        "implication": "Watch for breakouts through the gamma flip — they tend to accelerate quickly.",
        "beginner_emoji": "🎯",
    },
    "call_wall": {
        "term": "Call Wall",
        "category": "gex",
        "simple": "The strike price with the most call options — acts as a strong resistance level.",
        "practical": "Dealers are heavily short calls here. They sell stock as price approaches to hedge.",
        "implication": "Expect price to stall or reverse at the call wall. Use it as a target or exit zone.",
        "beginner_emoji": "🧱",
    },
    "put_wall": {
        "term": "Put Wall",
        "category": "gex",
        "simple": "The strike price with the most put options — acts as a strong support level.",
        "practical": "Dealers are short puts here. They buy stock as price falls to hedge — provides floor.",
        "implication": "Strong buying support near the put wall. Good level for long entries with tight stops.",
        "beginner_emoji": "🛡️",
    },
    "dex": {
        "term": "DEX — Delta Exposure",
        "category": "gex",
        "simple": "Measures how many shares dealers need to buy or sell to stay hedged.",
        "practical": "Positive DEX = dealers own shares and will sell on rallies. Negative = they'll buy dips.",
        "implication": "Negative DEX is bullish — mechanical buying support on pullbacks.",
        "beginner_emoji": "📊",
    },
    "vex": {
        "term": "VEX — Vanna Exposure",
        "category": "gex",
        "simple": "Measures how dealer hedging changes when volatility (VIX) moves.",
        "practical": "When VIX drops, positive VEX forces dealers to buy shares — can fuel rallies.",
        "implication": "Large VEX + falling VIX = mechanical tailwind for longs. Watch VIX carefully.",
        "beginner_emoji": "🌊",
    },
    "gamma_squeeze": {
        "term": "Gamma Squeeze",
        "category": "gex",
        "simple": "A fast, self-reinforcing price spike caused by dealer hedging pressure.",
        "practical": "As price rises, dealers buy more stock to hedge calls → pushes price higher → repeat.",
        "implication": "In negative GEX with strong upward momentum — join the move early, target is far.",
        "beginner_emoji": "🚀",
    },
    "positive_gex": {
        "term": "Positive Gamma Regime",
        "category": "gex",
        "simple": "Dealers are net long gamma — they act as a market stabilizer.",
        "practical": "Dealers sell when price rises, buy when it falls. Creates a 'magnetic' pinning effect.",
        "implication": "Avoid momentum plays. Trade mean reversion and fade moves at extreme levels.",
        "beginner_emoji": "📌",
    },
    "negative_gex": {
        "term": "Negative Gamma Regime",
        "category": "gex",
        "simple": "Dealers are net short gamma — they amplify market moves in both directions.",
        "practical": "Dealers buy as price rises and sell as price falls, making moves faster and bigger.",
        "implication": "Trend-following works well. Set wider stops. Moves can be explosive.",
        "beginner_emoji": "💥",
    },

    # ── Signals ───────────────────────────────────────────────────────────────

    "signal_score": {
        "term": "Signal Score",
        "category": "signal",
        "simple": "A 0–100 rating of how strong a trade setup is right now.",
        "practical": "Combines cycle timing, GEX, momentum, volatility, and news into one number.",
        "implication": "Score above 72 = high confidence. 52–72 = medium. Below 52 = skip the trade.",
        "beginner_emoji": "⭐",
    },
    "confidence_high": {
        "term": "High Confidence Signal",
        "category": "signal",
        "simple": "Multiple indicators align strongly — the setup has the best odds.",
        "practical": "Score above 72. At least 4 factors agree on direction.",
        "implication": "This is when to size up. Full position allocation applies.",
        "beginner_emoji": "✅",
    },
    "confidence_medium": {
        "term": "Medium Confidence Signal",
        "category": "signal",
        "simple": "The setup looks good but some factors are missing or neutral.",
        "practical": "Score 52–72. Proceed with smaller position or wait for confirmation.",
        "implication": "Half size recommended. Watch for confirmation candle before entry.",
        "beginner_emoji": "⚠️",
    },
    "signal_long": {
        "term": "Long Signal",
        "category": "signal",
        "simple": "The system thinks price is more likely to go UP.",
        "practical": "Buy the stock/ETF and profit if it rises. Exit at target or stop.",
        "implication": "Enter near current price. Stop below support. Target at resistance.",
        "beginner_emoji": "📈",
    },
    "signal_short": {
        "term": "Short Signal",
        "category": "signal",
        "simple": "The system thinks price is more likely to go DOWN.",
        "practical": "Sell (or buy put options) and profit if it falls. Not available in all accounts.",
        "implication": "Inverse ETFs (SQQQ, SPXU) let you profit from shorts without margin account.",
        "beginner_emoji": "📉",
    },

    # ── Technical Indicators ──────────────────────────────────────────────────

    "rsi": {
        "term": "RSI — Relative Strength Index",
        "category": "indicator",
        "simple": "Measures if a stock is overbought (too high) or oversold (too low).",
        "practical": "RSI above 70 = likely overbought. RSI below 30 = likely oversold.",
        "implication": "Don't buy at RSI 80. Wait for pullback. RSI 30 near support = strong buy zone.",
        "beginner_emoji": "🌡️",
    },
    "macd": {
        "term": "MACD — Moving Average Convergence Divergence",
        "category": "indicator",
        "simple": "Shows when a trend is gaining or losing strength.",
        "practical": "MACD crossing above zero = bullish momentum. Crossing below = bearish.",
        "implication": "MACD crossover in direction of the trend = higher probability entry.",
        "beginner_emoji": "📡",
    },
    "atr": {
        "term": "ATR — Average True Range",
        "category": "indicator",
        "simple": "Measures how much a stock typically moves each day.",
        "practical": "High ATR = volatile stock with big moves. Low ATR = quiet, slow-moving stock.",
        "implication": "Use ATR to set stops. Stop = 1.5 × ATR below entry to avoid being shaken out.",
        "beginner_emoji": "📏",
    },
    "volume_ratio": {
        "term": "Volume Ratio",
        "category": "indicator",
        "simple": "Today's trading volume compared to the 20-day average.",
        "practical": "3× ratio means 3× normal volume — unusual activity. Big moves on high volume stick.",
        "implication": "Volume confirms moves. High ratio + price breakout = strong signal.",
        "beginner_emoji": "🔊",
    },

    # ── Cycle / Wave Phase ────────────────────────────────────────────────────

    "cycle_top": {
        "term": "Cycle Top",
        "category": "cycle",
        "simple": "A time-based zone where price is more likely to reverse downward.",
        "practical": "Based on historical cycles — price tends to peak every N days/weeks.",
        "implication": "Near a cycle top: reduce long exposure, tighten stops, watch for reversal signals.",
        "beginner_emoji": "⛰️",
    },
    "cycle_bottom": {
        "term": "Cycle Bottom",
        "category": "cycle",
        "simple": "A time-based zone where price is more likely to reverse upward.",
        "practical": "Historical patterns suggest price tends to trough here before recovering.",
        "implication": "Near a cycle bottom with oversold RSI = high-probability buy zone.",
        "beginner_emoji": "🏔️",
    },
    "cycle_alignment": {
        "term": "Cycle Alignment",
        "category": "cycle",
        "simple": "How well short-term and long-term cycles agree on direction.",
        "practical": "All cycles pointing up = strongest bull signal. Mixed = proceed with caution.",
        "implication": "Only take trades where cycles align with GEX and momentum for highest odds.",
        "beginner_emoji": "🎯",
    },

    # ── Market Regime ─────────────────────────────────────────────────────────

    "vix": {
        "term": "VIX — Fear & Greed Index",
        "category": "regime",
        "simple": "Measures how fearful or calm the market is right now.",
        "practical": "VIX below 15 = calm, trending market. VIX above 30 = fear, volatile swings.",
        "implication": "Low VIX: trend-follow. High VIX: mean-revert. VIX spike: look for reversal.",
        "beginner_emoji": "😱",
    },
    "market_regime": {
        "term": "Market Regime",
        "category": "regime",
        "simple": "The current 'mode' of the market — trending, choppy, or breakout.",
        "practical": "TRENDING: follow momentum. RANGE: trade bounces. BREAKOUT: catch expansion.",
        "implication": "Matching your strategy to the regime doubles your win rate.",
        "beginner_emoji": "🗺️",
    },
    "momentum": {
        "term": "Momentum",
        "category": "indicator",
        "simple": "How fast and strong the current price move is.",
        "practical": "Strong momentum = price moving decisively in one direction with growing volume.",
        "implication": "Don't fight strong momentum. Wait for confirmation of reversal before fading.",
        "beginner_emoji": "💨",
    },

    # ── Patterns ──────────────────────────────────────────────────────────────

    "gamma_flip_breakout": {
        "term": "Gamma Flip Breakout",
        "category": "pattern",
        "simple": "Price crossing the gamma flip level — one of the strongest signals in the system.",
        "practical": "Flip = market structure change. Dealers switch from stabilizing to amplifying.",
        "implication": "High conviction directional trade. Enter on the breakout candle close.",
        "beginner_emoji": "🔥",
    },
    "support": {
        "term": "Support Level",
        "category": "pattern",
        "simple": "A price where buyers historically step in and prevent further decline.",
        "practical": "Often near the put wall, prior lows, or high-volume zones.",
        "implication": "Buy near support with stop just below. Risk is well-defined.",
        "beginner_emoji": "🏋️",
    },
    "resistance": {
        "term": "Resistance Level",
        "category": "pattern",
        "simple": "A price where sellers historically overwhelm buyers and halt rallies.",
        "practical": "Often near the call wall, prior highs, or round numbers.",
        "implication": "Take profits near resistance. If it breaks cleanly = strong continuation signal.",
        "beginner_emoji": "🚧",
    },

    # ── Scanner ───────────────────────────────────────────────────────────────

    "premarket_gap": {
        "term": "Premarket Gap",
        "category": "signal",
        "simple": "Price opened significantly higher or lower than yesterday's close.",
        "practical": "Gaps above 3% on news often continue. Gaps on no news often fill.",
        "implication": "Gap + catalyst + high volume = potential momentum play. Gap alone = wait.",
        "beginner_emoji": "🎪",
    },
    "volume_spike": {
        "term": "Volume Spike",
        "category": "signal",
        "simple": "Trading volume suddenly much higher than normal — something is happening.",
        "practical": "5× average volume means institutions or insiders are active in the stock.",
        "implication": "Volume spike in direction of trend = strong confirmation. Trade with it, not against.",
        "beginner_emoji": "📢",
    },
    "catalyst": {
        "term": "Catalyst",
        "category": "signal",
        "simple": "A news event that fundamentally changes how traders value a stock.",
        "practical": "Earnings beats, FDA approvals, analyst upgrades — move stocks 10-50% instantly.",
        "implication": "High-score signals with a catalyst are the highest conviction trades.",
        "beginner_emoji": "⚗️",
    },
}


def get_tooltip(key: str) -> TooltipDef | None:
    """Return tooltip for a concept key. Case-insensitive lookup."""
    return TOOLTIPS.get(key.lower().replace(" ", "_").replace("-", "_"))


def get_all_tooltips(category: str | None = None) -> list[TooltipDef]:
    """Return all tooltips, optionally filtered by category."""
    items = list(TOOLTIPS.values())
    if category:
        items = [t for t in items if t["category"] == category]
    return items
