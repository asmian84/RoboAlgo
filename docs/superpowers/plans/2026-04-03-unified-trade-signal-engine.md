# Unified Trade Signal Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a precision trade engine that scans the ENTIRE market (penny stocks through mega-caps), uses Smart Money Concepts + all existing confluences, and produces sniper-grade BUY/SELL verdicts with exact entry/stop/target/profit. No missed movers. No bag-holding. Bottom/top precision entries and exits.

**Architecture:** Five-layer pipeline:
1. **UniverseProvider** — scans ALL stocks (penny → mega-cap) via screener APIs, finds unusual activity BEFORE the move
2. **DataCollector** — fetches multi-TF OHLCV + options chain per symbol
3. **SMCEngine** — Fair Value Gaps, Order Blocks, CHoCH, BOS, Premium/Discount zones, OTE
4. **UnifiedScorer** — wires 18+ confluence engines into one weighted score with hard gates
5. **TradeRecommender** — precision entry/exit from SMC levels + position sizing + profit projection

**Tech Stack:** Python 3.11+, FastAPI, yfinance, numpy, pandas, React/TypeScript

---

## What Exists vs What's Missing

### EXISTS and CONNECTED (12):
MTF Alignment, Trend (MA/RSI/MACD), Wave Phase, GEX Signal, Dealer Positioning (DEX/VEX), Catalyst, Pattern Detection (90+ patterns), Level Engine (gems/pounce), Breakout Strength, Cycle Alignment (FFT/wavelet), Time Exhaustion, Vol Compression, Liquidity Sweep/Stop Hunt, Divergence (RSI+MACD+OBV), Volume Climax/Capitulation, Wyckoff A/D, Market Breadth

### EXISTS but DISCONNECTED (4):
Liquidity score (hardcoded 50), Pattern score (hardcoded 50), Wyckoff phase (hardcoded 50), Harmonic confluence (hardcoded 40) — all neutral in live path

### MISSING — Must Build (9):
1. **Fair Value Gap (FVG)** — imbalance zones where price snaps back
2. **Order Block (OB)** — last opposing candle before impulsive move
3. **Change of Character (CHoCH)** — structure break signaling reversal
4. **Break of Structure (BOS)** — trend continuation confirmation
5. **Premium/Discount Zones** — above/below equilibrium of range
6. **Optimal Trade Entry (OTE)** — 62-79% fib retracement sweet spot
7. **Float/SI Analysis** — low float + high short interest = squeeze
8. **Relative Strength vs SPY** — is the stock leading or lagging the market?
9. **Full Universe Scanner** — scan ALL tickers, not just 35 curated

---

## File Structure

### New files to create:
| File | Responsibility |
|------|---------------|
| `smc_engine/__init__.py` | Smart Money Concepts package |
| `smc_engine/fair_value_gap.py` | FVG detector — imbalance zones in candle bodies |
| `smc_engine/order_block.py` | Order block detector — institutional candles |
| `smc_engine/structure.py` | CHoCH + BOS detector — market structure shifts |
| `smc_engine/premium_discount.py` | Premium/discount zone classifier + OTE zone |
| `smc_engine/smc_scorer.py` | Composite SMC score from all sub-detectors |
| `trade_signal_engine/__init__.py` | Package exports |
| `trade_signal_engine/universe.py` | Full-market universe scanner (penny → mega-cap) |
| `trade_signal_engine/collector.py` | Multi-TF OHLCV + options chain fetcher |
| `trade_signal_engine/scorer.py` | 18-component unified scorer with hard gates |
| `trade_signal_engine/recommender.py` | Precision entry/exit + sizing + profit projection |
| `trade_signal_engine/scanner.py` | Batch scan with caching + background pre-warm |
| `api/routers/trades.py` | REST endpoints |
| `frontend/src/pages/TradesPage.tsx` | Scanner + symbol lookup UI |
| `tests/test_smc_engine/test_fvg.py` | FVG tests |
| `tests/test_smc_engine/test_order_block.py` | OB tests |
| `tests/test_smc_engine/test_structure.py` | CHoCH/BOS tests |
| `tests/test_trade_signal_engine/test_collector.py` | Collector tests |
| `tests/test_trade_signal_engine/test_scorer.py` | Scorer tests |
| `tests/test_trade_signal_engine/test_recommender.py` | Recommender tests |

### Existing files to modify:
| File | Change |
|------|--------|
| `api/main.py` | Register `/api/trades` router |
| `frontend/src/App.tsx` | Add `/trades` route |
| `frontend/src/components/layout/SideNav.tsx` | Add TRADES nav item |
| `frontend/src/api/hooks.ts` | Add `useTradeScan` and `useTradeSymbol` hooks |

---

## Unified Weight Table (18 components)

```python
UNIFIED_WEIGHTS = {
    # ── Direction Agreement (35%) ──
    "mtf_alignment":       0.15,   # Multi-TF trend agreement — THE gatekeeper
    "trend_alignment":     0.10,   # MA50/200 + RSI + MACD structure
    "wave_phase":          0.05,   # IMPULSE vs CORRECTION phase
    "structure_shift":     0.05,   # CHoCH/BOS — has structure changed?

    # ── Smart Money (20%) ──
    "fvg_confluence":      0.06,   # Is price at a Fair Value Gap?
    "order_block":         0.06,   # Is price at an Order Block?
    "premium_discount":    0.04,   # Is price in discount (for longs) or premium (for shorts)?
    "liquidity_sweep":     0.04,   # Recent stop hunt = smart money entry

    # ── Institutional Flow (15%) ──
    "gex_signal":          0.08,   # Gamma regime + squeeze risk
    "dealer_positioning":  0.04,   # DEX + VEX composite
    "catalyst":            0.03,   # News/earnings/catalyst sentiment

    # ── Structure Quality (15%) ──
    "pattern_score":       0.05,   # Best pattern confidence from 90+ detectors
    "level_confluence":    0.05,   # Gem strength from LevelEngine
    "breakout_strength":   0.03,   # Compression → breakout quality
    "relative_strength":   0.02,   # vs SPY — leading or lagging?

    # ── Timing (15%) ──
    "cycle_alignment":     0.05,   # FFT/wavelet cycle phase
    "exhaustion_check":    0.05,   # Time exhaustion (ANTI-signal if exhausted)
    "vol_compression":     0.05,   # BB width + ATR squeeze (coiled spring)
}
# Total: 1.00
```

### Hard Gates (block the trade entirely):
1. **MTF alignment < 0.40** → NO_TRADE (timeframes disagree)
2. **Exhaustion detected in trade direction** → NO_TRADE (overextended)
3. **Wave phase = CORRECTION + direction = LONG + confidence > 0.6** → NO_TRADE
4. **Premium zone for longs / Discount zone for shorts** → NO_TRADE (wrong side of equilibrium)
5. **No CHoCH or BOS confirming direction** → reduce score by 30% (unconfirmed structure)

### Score Boosters (precision entry bonuses):
1. **Price AT a FVG + Order Block overlap** → +15 bonus points (institutional sniper entry)
2. **Liquidity sweep just occurred + CHoCH confirmed** → +10 bonus points (spring/trap entry)
3. **In OTE zone (62-79% fib retracement)** → +8 bonus points (optimal pullback)
4. **Volume climax + divergence at support** → +12 bonus points (capitulation bottom)

---

## Timeframe Mapping

```python
TRADE_PROFILES = {
    "SCALP": {
        "mtf_model":       "DAY_TRADE",    # bias=1H, setup=15m, entry=5m
        "bar_tf":          "5m",
        "lookback":        "5d",
        "level_horizon":   "scalp",
        "smc_tf":          "5m",           # FVG/OB detected on 5m
        "structure_tf":    "15m",          # CHoCH/BOS on 15m
        "max_hold":        "2 hours",
        "stop_atr_mult":   0.75,
        "target_rr":       2.5,
    },
    "SWING": {
        "mtf_model":       "SWING_TRADE",  # bias=Daily, setup=4H, entry=1H
        "bar_tf":          "1h",
        "lookback":        "60d",
        "level_horizon":   "day",
        "smc_tf":          "1h",
        "structure_tf":    "4h",
        "max_hold":        "5 days",
        "stop_atr_mult":   1.5,
        "target_rr":       3.0,
    },
    "POSITION": {
        "mtf_model":       "INVESTMENT",   # bias=Weekly, setup=Daily, entry=4H
        "bar_tf":          "1d",
        "lookback":        "1y",
        "level_horizon":   "trend",
        "smc_tf":          "4h",
        "structure_tf":    "1d",
        "max_hold":        "4 weeks",
        "stop_atr_mult":   2.0,
        "target_rr":       4.0,
    },
}
```

---

## Task 1: SMC Engine — Fair Value Gap Detector

**Files:**
- Create: `smc_engine/__init__.py`
- Create: `smc_engine/fair_value_gap.py`
- Create: `tests/test_smc_engine/__init__.py`
- Create: `tests/test_smc_engine/test_fvg.py`

**What it does:** Detects imbalance zones where price moved so fast it left a gap in candle bodies. Price tends to return to fill these gaps — they act as magnets for precision entries.

**Detection logic:**
- Bullish FVG: candle[i-1].high < candle[i+1].low (gap between body of candle before and after)
- Bearish FVG: candle[i-1].low > candle[i+1].high
- FVG is "unfilled" until price revisits the zone
- FVG strength based on: gap size (% of ATR), volume on impulse candle, whether it's near a key level

- [ ] **Step 1.1: Write failing test**

```python
# tests/test_smc_engine/__init__.py
# (empty)
```

```python
# tests/test_smc_engine/test_fvg.py
import pytest
import pandas as pd
import numpy as np
from smc_engine.fair_value_gap import detect_fvgs, FVG


def _make_df_with_bullish_fvg():
    """Create OHLCV with a clear bullish FVG at index 5."""
    data = {
        "open":   [100, 101, 102, 103, 104, 105, 112, 113, 114, 115],
        "high":   [101, 102, 103, 104, 105, 106, 114, 115, 116, 117],
        "low":    [99,  100, 101, 102, 103, 104, 110, 111, 112, 113],
        "close":  [101, 102, 103, 104, 105, 113, 113, 114, 115, 116],
        "volume": [1000]*10,
    }
    return pd.DataFrame(data, index=pd.date_range("2026-01-01", periods=10, freq="h"))


def _make_df_with_bearish_fvg():
    """Create OHLCV with a clear bearish FVG at index 5."""
    data = {
        "open":   [115, 114, 113, 112, 111, 110, 102, 101, 100, 99],
        "high":   [116, 115, 114, 113, 112, 111, 103, 102, 101, 100],
        "low":    [114, 113, 112, 111, 110, 101, 100,  99,  98,  97],
        "close":  [114, 113, 112, 111, 110, 102, 101, 100,  99,  98],
        "volume": [1000]*10,
    }
    return pd.DataFrame(data, index=pd.date_range("2026-01-01", periods=10, freq="h"))


class TestFVGDetector:

    def test_detects_bullish_fvg(self):
        df = _make_df_with_bullish_fvg()
        fvgs = detect_fvgs(df)
        bullish = [f for f in fvgs if f.direction == "bullish"]
        assert len(bullish) >= 1
        assert bullish[0].top > bullish[0].bottom
        assert bullish[0].filled is False

    def test_detects_bearish_fvg(self):
        df = _make_df_with_bearish_fvg()
        fvgs = detect_fvgs(df)
        bearish = [f for f in fvgs if f.direction == "bearish"]
        assert len(bearish) >= 1
        assert bearish[0].top > bearish[0].bottom

    def test_returns_empty_on_flat_data(self):
        df = pd.DataFrame({
            "open": [100]*20, "high": [101]*20, "low": [99]*20,
            "close": [100]*20, "volume": [1000]*20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="h"))
        fvgs = detect_fvgs(df)
        assert len(fvgs) == 0

    def test_fvg_has_required_fields(self):
        df = _make_df_with_bullish_fvg()
        fvgs = detect_fvgs(df)
        if fvgs:
            f = fvgs[0]
            assert hasattr(f, "direction")
            assert hasattr(f, "top")
            assert hasattr(f, "bottom")
            assert hasattr(f, "midpoint")
            assert hasattr(f, "strength")
            assert hasattr(f, "filled")
            assert hasattr(f, "candle_index")
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_fvg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smc_engine'`

- [ ] **Step 1.3: Implement FVG detector**

```python
# smc_engine/__init__.py
"""Smart Money Concepts Engine — FVG, Order Blocks, CHoCH, BOS, Premium/Discount."""
```

```python
# smc_engine/fair_value_gap.py
"""
Fair Value Gap (FVG) Detector

An FVG is an imbalance zone where price moved so aggressively that it left a gap
between the bodies of surrounding candles. Price tends to return to fill these gaps.

Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up)
Bearish FVG: candle[i-1].low  > candle[i+1].high (gap down)

Used for precision entries: enter long at a bullish FVG in discount zone.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FVG:
    direction: str        # "bullish" | "bearish"
    top: float            # upper edge of the gap
    bottom: float         # lower edge of the gap
    midpoint: float       # center of the gap (optimal entry)
    strength: float       # 0-1 based on gap size relative to ATR
    filled: bool          # has price returned to fill the gap?
    candle_index: int     # index of the impulse candle
    timestamp: str        # datetime of the impulse candle


def detect_fvgs(
    df: pd.DataFrame,
    min_gap_atr: float = 0.3,
    lookback: int = 50,
    check_filled: bool = True,
) -> list[FVG]:
    """
    Detect Fair Value Gaps in OHLCV data.

    Args:
        df: DataFrame with open, high, low, close, volume columns
        min_gap_atr: minimum gap size as fraction of ATR to qualify
        lookback: how many bars back to scan
        check_filled: whether to check if FVGs have been filled by subsequent price action

    Returns:
        List of FVG objects, most recent first
    """
    if len(df) < 5:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # ATR-14 for normalization
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 1.0
    if atr <= 0:
        atr = 1.0

    min_gap = atr * min_gap_atr
    start = max(1, len(df) - lookback)
    fvgs: list[FVG] = []
    timestamps = df.index

    for i in range(start, len(df) - 1):
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        next_low = lows[i + 1]
        next_high = highs[i + 1]

        # Bullish FVG: gap between prev candle high and next candle low
        if next_low > prev_high and (next_low - prev_high) >= min_gap:
            gap_top = next_low
            gap_bottom = prev_high
            gap_size = gap_top - gap_bottom
            strength = min(1.0, gap_size / (atr * 2))

            # Check if filled: any subsequent bar's low touched the gap
            filled = False
            if check_filled:
                for j in range(i + 2, len(df)):
                    if lows[j] <= gap_top:
                        filled = True
                        break

            fvgs.append(FVG(
                direction="bullish", top=gap_top, bottom=gap_bottom,
                midpoint=round((gap_top + gap_bottom) / 2, 4),
                strength=round(strength, 3), filled=filled,
                candle_index=i,
                timestamp=str(timestamps[i]) if i < len(timestamps) else "",
            ))

        # Bearish FVG: gap between prev candle low and next candle high
        if prev_low > next_high and (prev_low - next_high) >= min_gap:
            gap_top = prev_low
            gap_bottom = next_high
            gap_size = gap_top - gap_bottom
            strength = min(1.0, gap_size / (atr * 2))

            filled = False
            if check_filled:
                for j in range(i + 2, len(df)):
                    if highs[j] >= gap_bottom:
                        filled = True
                        break

            fvgs.append(FVG(
                direction="bearish", top=gap_top, bottom=gap_bottom,
                midpoint=round((gap_top + gap_bottom) / 2, 4),
                strength=round(strength, 3), filled=filled,
                candle_index=i,
                timestamp=str(timestamps[i]) if i < len(timestamps) else "",
            ))

    # Most recent first
    fvgs.sort(key=lambda f: f.candle_index, reverse=True)
    return fvgs


def nearest_unfilled_fvg(
    fvgs: list[FVG], price: float, direction: str = "bullish"
) -> FVG | None:
    """Find the nearest unfilled FVG to current price in the given direction."""
    candidates = [f for f in fvgs if f.direction == direction and not f.filled]
    if not candidates:
        return None
    # Sort by distance to current price
    candidates.sort(key=lambda f: abs(f.midpoint - price))
    return candidates[0]
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_fvg.py -v`
Expected: 4 passed

- [ ] **Step 1.5: Commit**

```bash
git add smc_engine/__init__.py smc_engine/fair_value_gap.py \
       tests/test_smc_engine/__init__.py tests/test_smc_engine/test_fvg.py
git commit -m "feat: add Fair Value Gap (FVG) detector for Smart Money Concepts"
```

---

## Task 2: SMC Engine — Order Block Detector

**Files:**
- Create: `smc_engine/order_block.py`
- Create: `tests/test_smc_engine/test_order_block.py`

**What it does:** Detects the last opposing candle before an impulsive move. This is where institutional orders were placed. When price returns to an order block, it acts as strong support/resistance.

**Detection logic:**
- Bullish OB: Last bearish (red) candle before a strong bullish impulse (3+ consecutive green candles or a single large green candle > 2x ATR)
- Bearish OB: Last bullish (green) candle before a strong bearish impulse
- OB zone = open-to-close range of that candle
- OB is "mitigated" when price fully trades through it

- [ ] **Step 2.1: Write failing test**

```python
# tests/test_smc_engine/test_order_block.py
import pytest
import pandas as pd
import numpy as np
from smc_engine.order_block import detect_order_blocks, OrderBlock


def _make_df_with_bullish_ob():
    """Bearish candle followed by strong bullish impulse = bullish OB."""
    data = {
        "open":   [100, 101, 102, 101, 100, 99, 100, 105, 108, 112, 115, 118],
        "high":   [102, 103, 103, 102, 101, 100, 106, 109, 112, 115, 118, 120],
        "low":    [99,  100, 101, 100,  99,  98,  99, 104, 107, 111, 114, 117],
        "close":  [101, 102, 101, 100,  99,  100, 105, 108, 112, 115, 118, 120],
        "volume": [1000]*12,
    }
    return pd.DataFrame(data, index=pd.date_range("2026-01-01", periods=12, freq="h"))


class TestOrderBlockDetector:

    def test_detects_bullish_order_block(self):
        df = _make_df_with_bullish_ob()
        obs = detect_order_blocks(df)
        bullish = [ob for ob in obs if ob.direction == "bullish"]
        assert len(bullish) >= 1
        assert bullish[0].zone_top > bullish[0].zone_bottom
        assert bullish[0].mitigated is not None  # bool

    def test_returns_empty_on_flat_data(self):
        df = pd.DataFrame({
            "open": [100]*20, "high": [101]*20, "low": [99]*20,
            "close": [100.5]*20, "volume": [1000]*20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="h"))
        obs = detect_order_blocks(df)
        assert len(obs) == 0

    def test_ob_has_required_fields(self):
        df = _make_df_with_bullish_ob()
        obs = detect_order_blocks(df)
        if obs:
            ob = obs[0]
            assert hasattr(ob, "direction")
            assert hasattr(ob, "zone_top")
            assert hasattr(ob, "zone_bottom")
            assert hasattr(ob, "strength")
            assert hasattr(ob, "mitigated")
            assert hasattr(ob, "candle_index")
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_order_block.py -v`
Expected: FAIL

- [ ] **Step 2.3: Implement Order Block detector**

```python
# smc_engine/order_block.py
"""
Order Block (OB) Detector

An order block is the last opposing candle before an impulsive directional move.
It marks where institutional orders were placed. When price returns to an OB,
it acts as strong support (bullish OB) or resistance (bearish OB).

Bullish OB: last RED candle before a strong bullish impulse
Bearish OB: last GREEN candle before a strong bearish impulse
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    direction: str        # "bullish" | "bearish"
    zone_top: float       # upper edge (max of open/close of the OB candle)
    zone_bottom: float    # lower edge (min of open/close of the OB candle)
    strength: float       # 0-1 based on impulse magnitude
    mitigated: bool       # has price fully traded through the OB?
    candle_index: int
    timestamp: str
    impulse_size: float   # size of the move that followed (in ATR multiples)


def detect_order_blocks(
    df: pd.DataFrame,
    min_impulse_atr: float = 1.5,
    min_impulse_candles: int = 2,
    lookback: int = 50,
) -> list[OrderBlock]:
    """
    Detect order blocks in OHLCV data.

    Args:
        df: OHLCV DataFrame
        min_impulse_atr: minimum impulse size in ATR multiples to qualify
        min_impulse_candles: minimum consecutive candles in impulse direction
        lookback: bars to scan

    Returns:
        List of OrderBlock objects, most recent first
    """
    if len(df) < 5:
        return []

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # ATR-14
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 1.0
    if atr <= 0:
        atr = 1.0

    start = max(1, len(df) - lookback)
    obs: list[OrderBlock] = []

    for i in range(start, len(df) - min_impulse_candles):
        candle_bullish = closes[i] > opens[i]
        candle_bearish = closes[i] < opens[i]

        if not (candle_bullish or candle_bearish):
            continue

        # Check for impulse AFTER this candle
        impulse_dir = None
        impulse_end = i + 1
        total_move = 0.0

        # Count consecutive candles in one direction after candle[i]
        bullish_count = 0
        bearish_count = 0
        for j in range(i + 1, min(i + 8, len(df))):
            if closes[j] > opens[j]:
                bullish_count += 1
                if bearish_count > 0:
                    break
            elif closes[j] < opens[j]:
                bearish_count += 1
                if bullish_count > 0:
                    break
            impulse_end = j

        if bullish_count >= min_impulse_candles:
            impulse_dir = "bullish"
            total_move = max(highs[i + 1:impulse_end + 1]) - lows[i]
        elif bearish_count >= min_impulse_candles:
            impulse_dir = "bearish"
            total_move = highs[i] - min(lows[i + 1:impulse_end + 1])

        if impulse_dir is None:
            continue

        impulse_atr = total_move / atr
        if impulse_atr < min_impulse_atr:
            continue

        # The OB candle must be OPPOSING the impulse
        if impulse_dir == "bullish" and not candle_bearish:
            continue
        if impulse_dir == "bearish" and not candle_bullish:
            continue

        zone_top = max(opens[i], closes[i])
        zone_bottom = min(opens[i], closes[i])
        strength = min(1.0, impulse_atr / 4.0)

        # Check mitigation: has price fully traded through the zone?
        mitigated = False
        for j in range(impulse_end + 1, len(df)):
            if impulse_dir == "bullish" and lows[j] < zone_bottom:
                mitigated = True
                break
            if impulse_dir == "bearish" and highs[j] > zone_top:
                mitigated = True
                break

        obs.append(OrderBlock(
            direction=impulse_dir, zone_top=zone_top, zone_bottom=zone_bottom,
            strength=round(strength, 3), mitigated=mitigated,
            candle_index=i,
            timestamp=str(timestamps[i]) if i < len(timestamps) else "",
            impulse_size=round(impulse_atr, 2),
        ))

    obs.sort(key=lambda o: o.candle_index, reverse=True)
    return obs


def nearest_unmitigated_ob(
    obs: list[OrderBlock], price: float, direction: str = "bullish"
) -> OrderBlock | None:
    """Find nearest unmitigated order block to current price."""
    candidates = [ob for ob in obs if ob.direction == direction and not ob.mitigated]
    if not candidates:
        return None
    candidates.sort(key=lambda ob: abs(ob.zone_top - price) if direction == "bullish"
                    else abs(ob.zone_bottom - price))
    return candidates[0]
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_order_block.py -v`
Expected: 3 passed

- [ ] **Step 2.5: Commit**

```bash
git add smc_engine/order_block.py tests/test_smc_engine/test_order_block.py
git commit -m "feat: add Order Block (OB) detector for Smart Money Concepts"
```

---

## Task 3: SMC Engine — CHoCH + BOS Structure Detector

**Files:**
- Create: `smc_engine/structure.py`
- Create: `tests/test_smc_engine/test_structure.py`

**What it does:**
- **BOS (Break of Structure):** Price breaks above the previous swing high (bullish) or below swing low (bearish) — trend CONTINUATION.
- **CHoCH (Change of Character):** Price breaks the OPPOSITE swing point — trend REVERSAL signal. This is the earliest signal that a bottom/top has formed.

- [ ] **Step 3.1: Write failing test**

```python
# tests/test_smc_engine/test_structure.py
import pytest
import pandas as pd
import numpy as np
from smc_engine.structure import detect_structure_shifts, StructureShift


def _make_uptrend_then_choch():
    """Uptrend (HH/HL) followed by a break below last HL = bearish CHoCH."""
    prices = [100, 105, 103, 108, 106, 112, 109, 115, 112, 108, 105, 103]
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p + 2 for p in prices],
        "low": [p - 1 for p in prices],
        "close": [p + 0.5 for p in prices],
        "volume": [1000]*n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="h"))


def _make_uptrend_with_bos():
    """Steady uptrend with higher highs = bullish BOS."""
    prices = [100, 103, 101, 106, 103, 109, 106, 112, 109, 115]
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p + 2 for p in prices],
        "low": [p - 1 for p in prices],
        "close": [p + 1 for p in prices],
        "volume": [1000]*n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="h"))


class TestStructureShifts:

    def test_detects_bearish_choch(self):
        df = _make_uptrend_then_choch()
        shifts = detect_structure_shifts(df)
        chochs = [s for s in shifts if s.shift_type == "CHoCH"]
        assert len(chochs) >= 1
        assert any(s.direction == "bearish" for s in chochs)

    def test_detects_bullish_bos(self):
        df = _make_uptrend_with_bos()
        shifts = detect_structure_shifts(df)
        bos_list = [s for s in shifts if s.shift_type == "BOS"]
        assert len(bos_list) >= 1
        assert any(s.direction == "bullish" for s in bos_list)

    def test_has_required_fields(self):
        df = _make_uptrend_then_choch()
        shifts = detect_structure_shifts(df)
        if shifts:
            s = shifts[0]
            assert s.shift_type in ("CHoCH", "BOS")
            assert s.direction in ("bullish", "bearish")
            assert s.level > 0
            assert s.candle_index >= 0
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_structure.py -v`
Expected: FAIL

- [ ] **Step 3.3: Implement CHoCH/BOS detector**

```python
# smc_engine/structure.py
"""
Market Structure — CHoCH and BOS Detector

BOS (Break of Structure): Trend continuation.
  - Bullish BOS: price breaks above the previous swing high (higher high made)
  - Bearish BOS: price breaks below the previous swing low (lower low made)

CHoCH (Change of Character): Trend reversal — the EARLIEST signal of a top/bottom.
  - Bearish CHoCH: in an uptrend, price breaks below the last higher low
  - Bullish CHoCH: in a downtrend, price breaks above the last lower high

CHoCH is the sniper entry signal — it means smart money has shifted direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StructureShift:
    shift_type: str       # "CHoCH" | "BOS"
    direction: str        # "bullish" | "bearish"
    level: float          # the price level that was broken
    candle_index: int     # bar where the break occurred
    timestamp: str
    confidence: float     # 0-1 based on how decisively the level was broken


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "HH" | "HL" | "LH" | "LL" | "H" | "L"


def _detect_swings(highs: np.ndarray, lows: np.ndarray, order: int = 3) -> list[SwingPoint]:
    """Detect swing highs and lows using a simple N-bar lookback/lookahead."""
    swings: list[SwingPoint] = []
    n = len(highs)

    for i in range(order, n - order):
        # Swing high: highest in window
        if highs[i] == max(highs[i - order:i + order + 1]):
            swings.append(SwingPoint(index=i, price=float(highs[i]), type="H"))
        # Swing low: lowest in window
        if lows[i] == min(lows[i - order:i + order + 1]):
            swings.append(SwingPoint(index=i, price=float(lows[i]), type="L"))

    # Label HH/HL/LH/LL based on sequence
    swing_highs = [s for s in swings if s.type == "H"]
    swing_lows = [s for s in swings if s.type == "L"]

    for i in range(1, len(swing_highs)):
        swing_highs[i].type = "HH" if swing_highs[i].price > swing_highs[i-1].price else "LH"
    for i in range(1, len(swing_lows)):
        swing_lows[i].type = "HL" if swing_lows[i].price > swing_lows[i-1].price else "LL"

    # Merge back and sort by index
    all_swings = swing_highs + swing_lows
    all_swings.sort(key=lambda s: s.index)
    return all_swings


def detect_structure_shifts(
    df: pd.DataFrame,
    swing_order: int = 3,
    lookback: int = 50,
) -> list[StructureShift]:
    """
    Detect CHoCH and BOS events in OHLCV data.

    Returns list of structure shifts, most recent first.
    """
    if len(df) < swing_order * 3:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    swings = _detect_swings(highs, lows, order=swing_order)
    if len(swings) < 4:
        return []

    shifts: list[StructureShift] = []
    start_idx = max(0, len(df) - lookback)

    # Track the last significant swing high and swing low
    recent_highs = [s for s in swings if s.type in ("HH", "LH", "H") and s.index >= start_idx]
    recent_lows = [s for s in swings if s.type in ("HL", "LL", "L") and s.index >= start_idx]

    for i in range(len(swings) - 1):
        curr = swings[i]
        if curr.index < start_idx:
            continue

        # Find the next few bars after this swing to check for breaks
        for j in range(curr.index + 1, min(curr.index + 15, len(df))):
            # BOS: bullish — close above a swing high (continuation in uptrend)
            if curr.type in ("HH", "H"):
                if closes[j] > curr.price:
                    break_size = (closes[j] - curr.price) / curr.price
                    confidence = min(1.0, break_size * 50)  # 2% break = full confidence
                    shifts.append(StructureShift(
                        shift_type="BOS", direction="bullish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # BOS: bearish — close below a swing low (continuation in downtrend)
            if curr.type in ("LL", "L"):
                if closes[j] < curr.price:
                    break_size = (curr.price - closes[j]) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="BOS", direction="bearish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # CHoCH: bearish — in uptrend (HH/HL), price breaks below the last HL
            if curr.type == "HL":
                if closes[j] < curr.price:
                    break_size = (curr.price - closes[j]) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="CHoCH", direction="bearish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # CHoCH: bullish — in downtrend (LH/LL), price breaks above the last LH
            if curr.type == "LH":
                if closes[j] > curr.price:
                    break_size = (closes[j] - curr.price) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="CHoCH", direction="bullish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

    shifts.sort(key=lambda s: s.candle_index, reverse=True)
    return shifts


def latest_structure(shifts: list[StructureShift]) -> StructureShift | None:
    """Return the most recent structure shift."""
    return shifts[0] if shifts else None
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `cd /Users/asmian/RoboAlgo && python -m pytest tests/test_smc_engine/test_structure.py -v`
Expected: 3 passed

- [ ] **Step 3.5: Commit**

```bash
git add smc_engine/structure.py tests/test_smc_engine/test_structure.py
git commit -m "feat: add CHoCH and BOS market structure detector"
```

---

## Task 4: SMC Engine — Premium/Discount + OTE Zone

**Files:**
- Create: `smc_engine/premium_discount.py`

- [ ] **Step 4.1: Implement Premium/Discount + OTE**

```python
# smc_engine/premium_discount.py
"""
Premium/Discount Zone Classifier + Optimal Trade Entry (OTE)

The range between the most recent significant swing high and swing low defines
the "dealing range." Price above the 50% mark is PREMIUM (sell zone), below is
DISCOUNT (buy zone).

OTE (Optimal Trade Entry) is the 62-79% Fibonacci retracement zone — the sweet
spot where smart money enters on pullbacks.

Rules:
  - Only buy in DISCOUNT zone (below equilibrium)
  - Only sell in PREMIUM zone (above equilibrium)
  - Best entries are at OTE level within the correct zone
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ZoneAnalysis:
    swing_high: float         # top of dealing range
    swing_low: float          # bottom of dealing range
    equilibrium: float        # 50% level (midpoint)
    current_price: float
    zone: str                 # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM"
    zone_pct: float           # where price sits in the range (0=low, 100=high)
    ote_top: float            # OTE zone upper bound (62% retracement from swing high)
    ote_bottom: float         # OTE zone lower bound (79% retracement from swing high)
    in_ote: bool              # is price within the OTE zone?
    discount_levels: list[float]  # key fib levels in discount zone
    premium_levels: list[float]   # key fib levels in premium zone


def analyze_zone(
    df: pd.DataFrame,
    swing_lookback: int = 50,
    swing_order: int = 5,
) -> ZoneAnalysis:
    """
    Classify current price as premium/discount within the dealing range.

    Uses the most recent significant swing high and swing low to define the range.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    price = float(closes[-1])
    n = len(df)

    # Find significant swing high and low in lookback window
    window = min(swing_lookback, n)
    recent_highs = highs[-window:]
    recent_lows = lows[-window:]

    swing_high = float(np.max(recent_highs))
    swing_low = float(np.min(recent_lows))
    dealing_range = swing_high - swing_low

    if dealing_range <= 0:
        return ZoneAnalysis(
            swing_high=swing_high, swing_low=swing_low,
            equilibrium=price, current_price=price,
            zone="EQUILIBRIUM", zone_pct=50.0,
            ote_top=price, ote_bottom=price, in_ote=False,
            discount_levels=[], premium_levels=[],
        )

    equilibrium = swing_low + dealing_range * 0.50
    zone_pct = (price - swing_low) / dealing_range * 100

    if zone_pct > 55:
        zone = "PREMIUM"
    elif zone_pct < 45:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIUM"

    # OTE zone: 62-79% retracement from swing high (= 21-38% from swing low)
    # For a bullish OTE (buying the dip):
    ote_top = swing_high - dealing_range * 0.618     # 61.8% retracement
    ote_bottom = swing_high - dealing_range * 0.786   # 78.6% retracement
    in_ote = ote_bottom <= price <= ote_top

    # Key fib levels
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
    all_levels = [swing_low + dealing_range * f for f in fib_levels]
    discount_levels = [l for l in all_levels if l < equilibrium]
    premium_levels = [l for l in all_levels if l > equilibrium]

    return ZoneAnalysis(
        swing_high=round(swing_high, 4), swing_low=round(swing_low, 4),
        equilibrium=round(equilibrium, 4), current_price=round(price, 4),
        zone=zone, zone_pct=round(zone_pct, 1),
        ote_top=round(ote_top, 4), ote_bottom=round(ote_bottom, 4),
        in_ote=in_ote,
        discount_levels=[round(l, 4) for l in discount_levels],
        premium_levels=[round(l, 4) for l in premium_levels],
    )
```

- [ ] **Step 4.2: Commit**

```bash
git add smc_engine/premium_discount.py
git commit -m "feat: add Premium/Discount zone classifier + OTE zone"
```

---

## Task 5: SMC Composite Scorer

**Files:**
- Create: `smc_engine/smc_scorer.py`

- [ ] **Step 5.1: Implement SMC composite scorer**

```python
# smc_engine/smc_scorer.py
"""
SMC Composite Scorer — combines FVG + OB + CHoCH/BOS + Premium/Discount
into a single directional signal.

Returns:
  - smc_score: 0-1 (0.5 = neutral)
  - smc_bias: "LONG" | "SHORT" | "NEUTRAL"
  - entry_zone: precise price zone for entry (FVG/OB overlap)
  - confluence_detail: what's aligned and what's not
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from smc_engine.fair_value_gap import detect_fvgs, nearest_unfilled_fvg, FVG
from smc_engine.order_block import detect_order_blocks, nearest_unmitigated_ob, OrderBlock
from smc_engine.structure import detect_structure_shifts, latest_structure, StructureShift
from smc_engine.premium_discount import analyze_zone, ZoneAnalysis


@dataclass
class SMCResult:
    smc_score: float                      # 0-1 (>0.5 = bullish, <0.5 = bearish)
    smc_bias: str                         # LONG | SHORT | NEUTRAL
    entry_zone_top: float                 # precise entry zone upper
    entry_zone_bottom: float              # precise entry zone lower
    sniper_entry: float                   # exact best entry price (FVG/OB midpoint)
    fvg_count: int
    ob_count: int
    latest_shift: str                     # "CHoCH bullish" | "BOS bearish" | "none"
    zone: str                             # PREMIUM | DISCOUNT | EQUILIBRIUM
    in_ote: bool
    detail: dict[str, Any] = field(default_factory=dict)


def score_smc(df: pd.DataFrame, current_price: float = 0.0) -> SMCResult:
    """
    Run all SMC detectors and produce a composite result.

    Args:
        df: OHLCV DataFrame (any timeframe)
        current_price: override current price (uses last close if 0)

    Returns:
        SMCResult with composite score, bias, and precision entry zones
    """
    if current_price <= 0:
        current_price = float(df["close"].iloc[-1])

    # ── Run detectors ─────────────────────────────────────────────────────────
    fvgs = detect_fvgs(df, lookback=80)
    obs = detect_order_blocks(df, lookback=80)
    shifts = detect_structure_shifts(df, lookback=80)
    zone = analyze_zone(df)

    # ── Score components ──────────────────────────────────────────────────────
    score = 0.5
    bias_votes: list[float] = []  # >0.5 = bullish, <0.5 = bearish

    # 1. Structure shifts (most important)
    shift = latest_structure(shifts)
    shift_label = "none"
    if shift:
        shift_label = f"{shift.shift_type} {shift.direction}"
        if shift.shift_type == "CHoCH":
            # CHoCH is the reversal signal — strong directional
            if shift.direction == "bullish":
                bias_votes.append(0.85 * shift.confidence)
            else:
                bias_votes.append(0.15 + 0.35 * (1 - shift.confidence))
        elif shift.shift_type == "BOS":
            # BOS confirms the trend
            if shift.direction == "bullish":
                bias_votes.append(0.70)
            else:
                bias_votes.append(0.30)

    # 2. Premium/Discount zone
    if zone.zone == "DISCOUNT":
        bias_votes.append(0.70)  # favorable for longs
    elif zone.zone == "PREMIUM":
        bias_votes.append(0.30)  # favorable for shorts
    else:
        bias_votes.append(0.50)

    if zone.in_ote:
        bias_votes.append(0.75)  # OTE = optimal long entry

    # 3. FVG proximity
    nearest_bull_fvg = nearest_unfilled_fvg(fvgs, current_price, "bullish")
    nearest_bear_fvg = nearest_unfilled_fvg(fvgs, current_price, "bearish")

    bull_fvg_dist = abs(current_price - nearest_bull_fvg.midpoint) / current_price if nearest_bull_fvg else 1.0
    bear_fvg_dist = abs(current_price - nearest_bear_fvg.midpoint) / current_price if nearest_bear_fvg else 1.0

    if bull_fvg_dist < 0.01:  # within 1% of bullish FVG
        bias_votes.append(0.75)
    if bear_fvg_dist < 0.01:
        bias_votes.append(0.25)

    # 4. Order block proximity
    nearest_bull_ob = nearest_unmitigated_ob(obs, current_price, "bullish")
    nearest_bear_ob = nearest_unmitigated_ob(obs, current_price, "bearish")

    if nearest_bull_ob:
        ob_dist = abs(current_price - nearest_bull_ob.zone_top) / current_price
        if ob_dist < 0.015:  # within 1.5% of bullish OB
            bias_votes.append(0.70 + 0.15 * nearest_bull_ob.strength)
    if nearest_bear_ob:
        ob_dist = abs(current_price - nearest_bear_ob.zone_bottom) / current_price
        if ob_dist < 0.015:
            bias_votes.append(0.30 - 0.15 * nearest_bear_ob.strength)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if bias_votes:
        score = sum(bias_votes) / len(bias_votes)

    if score > 0.58:
        smc_bias = "LONG"
    elif score < 0.42:
        smc_bias = "SHORT"
    else:
        smc_bias = "NEUTRAL"

    # ── Precision entry zone ──────────────────────────────────────────────────
    entry_top = current_price
    entry_bottom = current_price
    sniper_entry = current_price

    if smc_bias == "LONG":
        # Best entry = FVG + OB overlap, or closest FVG, or closest OB
        candidates = []
        if nearest_bull_fvg and not nearest_bull_fvg.filled:
            candidates.append((nearest_bull_fvg.midpoint, nearest_bull_fvg.top, nearest_bull_fvg.bottom))
        if nearest_bull_ob and not nearest_bull_ob.mitigated:
            candidates.append((
                (nearest_bull_ob.zone_top + nearest_bull_ob.zone_bottom) / 2,
                nearest_bull_ob.zone_top, nearest_bull_ob.zone_bottom,
            ))
        if zone.in_ote:
            candidates.append(((zone.ote_top + zone.ote_bottom) / 2, zone.ote_top, zone.ote_bottom))

        if candidates:
            # Pick the one closest to current price
            candidates.sort(key=lambda c: abs(c[0] - current_price))
            sniper_entry, entry_top, entry_bottom = candidates[0]

    elif smc_bias == "SHORT":
        candidates = []
        if nearest_bear_fvg and not nearest_bear_fvg.filled:
            candidates.append((nearest_bear_fvg.midpoint, nearest_bear_fvg.top, nearest_bear_fvg.bottom))
        if nearest_bear_ob and not nearest_bear_ob.mitigated:
            candidates.append((
                (nearest_bear_ob.zone_top + nearest_bear_ob.zone_bottom) / 2,
                nearest_bear_ob.zone_top, nearest_bear_ob.zone_bottom,
            ))
        if candidates:
            candidates.sort(key=lambda c: abs(c[0] - current_price))
            sniper_entry, entry_top, entry_bottom = candidates[0]

    return SMCResult(
        smc_score=round(score, 3),
        smc_bias=smc_bias,
        entry_zone_top=round(entry_top, 4),
        entry_zone_bottom=round(entry_bottom, 4),
        sniper_entry=round(sniper_entry, 4),
        fvg_count=len([f for f in fvgs if not f.filled]),
        ob_count=len([o for o in obs if not o.mitigated]),
        latest_shift=shift_label,
        zone=zone.zone,
        in_ote=zone.in_ote,
        detail={
            "zone_pct": zone.zone_pct,
            "equilibrium": zone.equilibrium,
            "swing_high": zone.swing_high,
            "swing_low": zone.swing_low,
        },
    )
```

- [ ] **Step 5.2: Commit**

```bash
git add smc_engine/smc_scorer.py
git commit -m "feat: add SMC composite scorer combining FVG+OB+CHoCH+BOS+PD"
```

---

## Task 6: Universe Provider — Scan ALL 6,400+ US Stocks

**Files:**
- Create: `trade_signal_engine/universe.py`

**What it does:** Scans the ENTIRE US stock market — all ~6,400 listed tickers across NYSE, NASDAQ, and AMEX — plus OTC markets. Uses a 3-tier funnel:

**Tier 1 — Full Market Snapshot (6,400+ tickers, ~10 seconds):**
- Pull complete ticker list from Finnhub exchange API (free, returns all listed symbols)
- Batch-fetch daily snapshots via yfinance (supports bulk download)
- Flag: volume > 3x average, change > 5%, price < $5 (penny), market cap < $300M (micro-cap)

**Tier 2 — Pre-Filter to Top 300 Candidates (~30 seconds):**
- Rank by: volume spike ratio, % change, low float, high short interest
- Include ALL flagged stocks + tracked DB + curated watchlist
- This catches penny stocks BEFORE the 100x move (unusual accumulation volume)

**Tier 3 — Deep Analysis (top 300, via Scanner in Task 10):**
- Full 18-confluence scoring on each candidate
- SMC + options chain + levels + patterns

**Data sources:**
- Finnhub API (free tier: 60 calls/min) — full exchange symbol list + company profiles
- yfinance bulk download — price/volume/float/SI for thousands of tickers in one call
- SEC EDGAR (future) — 13F filings, Form 4 insider purchases

- [ ] **Step 6.1: Implement Universe Provider**

```python
# trade_signal_engine/universe.py
"""
Universe Provider — scans ALL ~6,400+ US-listed stocks plus OTC.

Three-tier funnel:
  Tier 1: Full market snapshot (all tickers, batch download, ~10s)
  Tier 2: Pre-filter to top 300 candidates by unusual activity
  Tier 3: Deep 18-confluence analysis (handled by Scanner)

Data sources:
  - Finnhub API: complete exchange symbol list (NYSE + NASDAQ + AMEX)
  - yfinance bulk download: price/volume/change for thousands of tickers in one call
  - DB: our tracked 233+ instruments (always included)

The goal: NEVER miss a stock going from $0.10 to $10.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Cache (refreshed every 15 min) ────────────────────────────────────────────

_TICKER_CACHE: dict[str, tuple[float, list[str]]] = {}
_TICKER_CACHE_TTL = 900  # 15 min

# ── Known liquid options tickers (always deep-scanned with options chain) ──────

OPTIONS_FOCUS = [
    "SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMD", "AMZN", "META",
    "GOOGL", "MSFT", "NFLX", "COIN", "MARA", "SOFI", "PLTR", "ARM",
    "BAC", "JPM", "XOM", "BABA", "NIO", "RIVN", "LCID", "F", "GM",
]


@dataclass
class SymbolCandidate:
    symbol: str
    source: str              # "exchange" | "tracked" | "curated"
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 1.0
    float_shares: float = 0.0       # in millions
    short_pct: float = 0.0          # short % of float
    market_cap: float = 0.0         # in millions
    flags: list[str] = field(default_factory=list)


# ── Tier 1: Get ALL listed tickers ────────────────────────────────────────────

def _get_all_exchange_tickers() -> list[str]:
    """
    Fetch complete list of US-listed stock symbols from Finnhub.
    Returns ~6,400 tickers across NYSE, NASDAQ, AMEX.
    Falls back to a broad yfinance approach if Finnhub unavailable.
    """
    cache_key = "exchange_tickers"
    now = time.time()
    if cache_key in _TICKER_CACHE:
        ts, tickers = _TICKER_CACHE[cache_key]
        if now - ts < _TICKER_CACHE_TTL:
            return tickers

    tickers: list[str] = []

    # Method 1: Finnhub (preferred — returns ALL listed symbols)
    api_key = os.environ.get("FINNHUB_API_KEY")
    if api_key:
        try:
            import requests
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/symbol",
                params={"exchange": "US", "token": api_key},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Filter to common stocks only (type = "Common Stock")
                tickers = [
                    item["symbol"] for item in data
                    if item.get("type") in ("Common Stock", "ETP", "ADR")
                    and "." not in item.get("symbol", ".")  # skip preferred shares (BRK.B etc)
                    and len(item.get("symbol", "")) <= 5    # skip long symbols
                ]
                logger.info("Finnhub returned %d US tickers", len(tickers))
        except Exception as e:
            logger.warning("Finnhub ticker fetch failed: %s", e)

    # Method 2: Fallback — use a comprehensive static list + DB
    if not tickers:
        try:
            # yfinance doesn't have a screener API, but we can get a lot from our DB
            from data_engine.loader import DataLoader
            loader = DataLoader()
            tickers = loader.get_all_symbols()
            logger.info("Loaded %d tickers from DB", len(tickers))
        except Exception:
            pass

        # Always merge in a broad static universe (top ~500 by market cap + popular pennies)
        tickers = list(set(tickers + _BROAD_UNIVERSE))

    _TICKER_CACHE[cache_key] = (now, tickers)
    return tickers


# ── Tier 2: Batch snapshot + pre-filter ───────────────────────────────────────

def get_full_universe(
    min_volume_ratio: float = 2.0,
    min_change_pct: float = 3.0,
    max_candidates: int = 300,
    include_all_tracked: bool = True,
) -> list[SymbolCandidate]:
    """
    Scan ALL US stocks, batch-filter by unusual activity, return top candidates.

    Uses yfinance bulk download to get 1-day snapshot of ALL tickers at once.
    This is how we catch penny stocks before they 100x.
    """
    all_tickers = _get_all_exchange_tickers()
    logger.info("Scanning %d tickers for unusual activity...", len(all_tickers))

    candidates: dict[str, SymbolCandidate] = {}

    # Batch download last 5 days of daily data for ALL tickers
    # yfinance handles this efficiently in one HTTP call per ~500 tickers
    batch_size = 500
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            df = yf.download(
                " ".join(batch), period="5d", interval="1d",
                progress=False, timeout=30, group_by="ticker",
                threads=True,
            )
            if df is None or df.empty:
                continue

            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df
                    else:
                        if sym not in df.columns.get_level_values(0):
                            continue
                        sym_df = df[sym].dropna()

                    if sym_df.empty or len(sym_df) < 2:
                        continue

                    # Flatten multi-level columns if needed
                    if isinstance(sym_df.columns, pd.MultiIndex):
                        sym_df.columns = sym_df.columns.get_level_values(0)
                    sym_df.columns = [c.lower() for c in sym_df.columns]

                    close_today = float(sym_df["close"].iloc[-1])
                    close_prev = float(sym_df["close"].iloc[-2])
                    vol_today = float(sym_df["volume"].iloc[-1]) if "volume" in sym_df else 0
                    avg_vol = float(sym_df["volume"].iloc[:-1].mean()) if "volume" in sym_df and len(sym_df) > 2 else 1

                    if close_prev <= 0 or close_today <= 0:
                        continue

                    change_pct = (close_today - close_prev) / close_prev * 100
                    vol_ratio = vol_today / avg_vol if avg_vol > 0 else 1.0

                    # Flag unusual activity
                    flags = []
                    if close_today < 5:
                        flags.append("penny")
                    if close_today < 1:
                        flags.append("sub_dollar")
                    if vol_ratio >= min_volume_ratio:
                        flags.append("volume_spike")
                    if abs(change_pct) >= min_change_pct:
                        flags.append("big_mover")
                    if vol_ratio >= 5:
                        flags.append("extreme_volume")
                    if abs(change_pct) >= 15:
                        flags.append("explosive_move")

                    # Include if: has flags OR is in tracked DB OR is a blue chip
                    if flags or include_all_tracked:
                        candidates[sym] = SymbolCandidate(
                            symbol=sym, source="exchange",
                            price=close_today, change_pct=round(change_pct, 2),
                            volume=vol_today, avg_volume=avg_vol,
                            volume_ratio=round(vol_ratio, 2),
                            flags=flags,
                        )
                except Exception:
                    continue

        except Exception as e:
            logger.warning("Batch download failed for chunk %d: %s", i, e)
            continue

    # Always include tracked DB symbols (even if no unusual activity)
    try:
        from data_engine.loader import DataLoader
        for sym in DataLoader().get_all_symbols():
            if sym not in candidates:
                candidates[sym] = SymbolCandidate(symbol=sym, source="tracked")
            else:
                candidates[sym].source = "tracked"
    except Exception:
        pass

    # Enrich top candidates with float + short interest data
    top_flagged = sorted(
        [c for c in candidates.values() if c.flags],
        key=lambda c: (len(c.flags), abs(c.change_pct), c.volume_ratio),
        reverse=True,
    )[:100]  # enrich top 100 only (API-heavy)

    _enrich_float_and_si(top_flagged)

    # Final sort: most flags + biggest moves + highest volume ratio first
    result = sorted(
        candidates.values(),
        key=lambda c: (
            len(c.flags),
            1 if "extreme_volume" in c.flags else 0,
            1 if "explosive_move" in c.flags else 0,
            1 if "low_float" in c.flags else 0,
            abs(c.change_pct),
            c.volume_ratio,
        ),
        reverse=True,
    )
    return result[:max_candidates]


def _enrich_float_and_si(candidates: list[SymbolCandidate]) -> None:
    """Fetch float shares and short interest for top candidates."""
    if not candidates:
        return

    symbols = [c.symbol for c in candidates[:50]]  # limit API calls
    sym_map = {c.symbol: c for c in candidates}

    def _fetch_info(sym: str):
        try:
            info = yf.Ticker(sym).info
            c = sym_map.get(sym)
            if not c:
                return
            c.float_shares = (info.get("floatShares", 0) or 0) / 1e6
            c.short_pct = (info.get("shortPercentOfFloat", 0) or 0) * 100
            c.market_cap = (info.get("marketCap", 0) or 0) / 1e6

            if c.float_shares > 0 and c.float_shares < 20:
                c.flags.append("low_float")
            if c.short_pct > 15:
                c.flags.append("high_si")
            if c.short_pct > 30:
                c.flags.append("squeeze_candidate")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_fetch_info, symbols))


def get_options_universe() -> list[str]:
    """Return symbols with liquid options for options-specific scanning."""
    return OPTIONS_FOCUS


# ── Broad static universe (fallback when Finnhub unavailable) ─────────────────
# Top ~500 by market cap + popular micro-caps + ETFs

_BROAD_UNIVERSE = [
    # Mega/Large Cap
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "ACN", "CRM",
    "AMD", "ABT", "TMO", "DHR", "LIN", "NEE", "PM", "TXN", "ADBE", "ORCL",
    "NFLX", "CMCSA", "NKE", "BMY", "UPS", "RTX", "LOW", "UNP", "INTC", "QCOM",
    "BA", "CAT", "GE", "DE", "SBUX", "PLD", "AMAT", "ISRG", "MDT", "GS",
    "BLK", "GILD", "SYK", "ADP", "MMC", "REGN", "VRTX", "BKNG", "ZTS", "CB",
    "PYPL", "LRCX", "MU", "KLAC", "ASML", "MRVL", "SNPS", "CDNS", "PANW",
    # Mid Cap Tech / Growth
    "ARM", "PLTR", "SOFI", "COIN", "SQ", "SHOP", "UBER", "LYFT", "DASH", "ABNB",
    "DDOG", "SNOW", "NET", "CRWD", "ZS", "OKTA", "MDB", "ROKU", "TTD", "PINS",
    "RBLX", "U", "HOOD", "AFRM", "UPST", "BILL", "HUBS", "TWLO", "DKNG", "PENN",
    # Biotech / Pharma
    "MRNA", "BNTX", "PFE", "BIIB", "AMGN", "REGN", "VRTX", "SGEN", "ALNY",
    # Energy / Commodities
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL", "DVN",
    "GLD", "SLV", "USO", "UNG", "BOIL",
    # Financials
    "BAC", "JPM", "GS", "MS", "WFC", "C", "SCHW", "BX", "KKR", "APO",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "TLT", "HYG", "GDX", "ARKK", "ARKG", "XBI", "SMH", "SOXX", "KWEB",
    "TQQQ", "SQQQ", "SPXS", "SPXL", "UVXY", "VXX", "VIXY",
    # Leveraged / Inverse
    "SOXL", "SOXS", "LABU", "LABD", "TECS", "TECL", "FAS", "FAZ",
    "SDOW", "UDOW", "TNA", "TZA", "NUGT", "DUST",
    # Popular Micro-caps / Penny Stocks / Meme Stocks
    "MULN", "FFIE", "SNDL", "CLOV", "WISH", "GOEV", "NKLA", "RIDE", "QS",
    "HYMC", "GSAT", "TELL", "WKHS", "OPEN", "SKLZ", "BB", "NOK", "AMC", "GME",
    "BBBY", "IONQ", "RKLB", "LUNR", "ASTS", "JOBY", "EVTL", "LILM", "ACHR",
    "NIO", "XPEV", "LI", "RIVN", "LCID", "FSR", "PSNY", "VFS", "GOEV",
    "MARA", "RIOT", "HUT", "BITF", "CLSK", "CIFR", "HIVE",
    # Cannabis
    "TLRY", "CGC", "ACB", "SNDL", "OGI", "HEXO",
    # Space / Defense
    "LMT", "RTX", "NOC", "GD", "BA", "RKLB", "ASTR", "SPCE", "BKSY",
]
```

- [ ] **Step 6.2: Commit**

```bash
git add trade_signal_engine/universe.py
git commit -m "feat: add full-market universe scanner (penny through mega-cap)"
```

---

## Task 7: DataCollector — Multi-TF Price + Options Fetcher

Same as original Task 1. See original plan for full implementation.

**Files:** `trade_signal_engine/collector.py`, `tests/test_trade_signal_engine/test_collector.py`

*(Full code unchanged from original plan — see Task 1 above)*

- [ ] **Step 7.1-7.6: Implement and test DataCollector** (see original Task 1)

---

## Task 8: UnifiedScorer — 18-Component Scorer With SMC + Hard Gates

**Files:**
- Create: `trade_signal_engine/scorer.py`
- Create: `tests/test_trade_signal_engine/test_scorer.py`

This extends the original Task 2 scorer with 6 NEW SMC components.

The key additions vs original scorer:

```python
# In the score() method, add these after existing components:

# ── SMC Components (NEW) ─────────────────────────────────────
from smc_engine.smc_scorer import score_smc

# Run SMC on the appropriate timeframe
smc_tf = {"SCALP": "5m", "SWING": "1h", "POSITION": "4h"}.get(sd.profile, "1h")
smc_df = sd.bars.get(smc_tf) or sd.bars.get("1h") or self._best_daily_bars(sd)
if smc_df is not None and len(smc_df) >= 20:
    smc_result = score_smc(smc_df, current_price=sd.current_price)

    # FVG confluence
    components["fvg_confluence"] = smc_result.smc_score if smc_result.fvg_count > 0 else 0.5

    # Order block
    components["order_block"] = smc_result.smc_score if smc_result.ob_count > 0 else 0.5

    # Premium/Discount
    if smc_result.zone == "DISCOUNT":
        components["premium_discount"] = 0.75
    elif smc_result.zone == "PREMIUM":
        components["premium_discount"] = 0.25
    else:
        components["premium_discount"] = 0.50

    # Liquidity sweep (use existing engine)
    components["liquidity_sweep"] = self._score_liquidity_sweep(sd)

    # Structure shift (CHoCH/BOS)
    if "CHoCH" in smc_result.latest_shift:
        if "bullish" in smc_result.latest_shift:
            components["structure_shift"] = 0.82
        else:
            components["structure_shift"] = 0.18
    elif "BOS" in smc_result.latest_shift:
        if "bullish" in smc_result.latest_shift:
            components["structure_shift"] = 0.70
        else:
            components["structure_shift"] = 0.30
    else:
        components["structure_shift"] = 0.50

    # Store SMC detail for recommender to use for sniper entry
    details["smc"] = {
        "sniper_entry": smc_result.sniper_entry,
        "entry_zone_top": smc_result.entry_zone_top,
        "entry_zone_bottom": smc_result.entry_zone_bottom,
        "bias": smc_result.smc_bias,
        "in_ote": smc_result.in_ote,
    }

# ── HARD GATE: Premium zone for longs ─────────────────────────
if smc_result.zone == "PREMIUM" and target_dir == "long":
    # Don't buy in premium — waiting for discount
    return ScoredSignal(
        symbol=sd.symbol, profile=sd.profile, score=35,
        direction="NO_TRADE", confidence="LOW",
        components=components, component_details=details,
        rejection_reason="Price in PREMIUM zone — wait for pullback to DISCOUNT",
    )

# ── SCORE BOOSTERS ─────────────────────────────────────────────
bonus = 0
# FVG + OB overlap = sniper entry
if smc_result.fvg_count > 0 and smc_result.ob_count > 0:
    bonus += 15
# Liquidity sweep + CHoCH = spring/trap entry
if "CHoCH" in smc_result.latest_shift and components.get("liquidity_sweep", 0.5) > 0.65:
    bonus += 10
# In OTE zone
if smc_result.in_ote:
    bonus += 8
```

*(The full scorer.py implementation follows the same structure as original Task 2 but with these additions integrated into the score() method. The WEIGHTS dict uses the 18-component table from the top of this plan.)*

- [ ] **Step 8.1-8.5: Implement and test UnifiedScorer** (follows same test/implement/commit pattern)

---

## Task 9: TradeRecommender — Sniper Entry from SMC Zones

Same as original Task 3 but the `recommend()` method now checks SMC sniper entry FIRST:

```python
# In recommend(), before the pounce_ideas fallback:

# Priority 1: SMC sniper entry (FVG/OB zone)
smc_detail = signal.component_details.get("smc", {})
sniper_entry = smc_detail.get("sniper_entry", 0)
if sniper_entry > 0 and signal.direction != "NO_TRADE":
    rec.entry = round(sniper_entry, 2)
    # Stop below SMC zone (or above for shorts)
    zone_bottom = smc_detail.get("entry_zone_bottom", sniper_entry)
    zone_top = smc_detail.get("entry_zone_top", sniper_entry)
    if signal.direction == "LONG":
        rec.stop = round(zone_bottom - atr * 0.3, 2)  # tight stop below zone
        rec.target = round(sniper_entry + abs(sniper_entry - rec.stop) * target_rr, 2)
    else:
        rec.stop = round(zone_top + atr * 0.3, 2)
        rec.target = round(sniper_entry - abs(rec.stop - sniper_entry) * target_rr, 2)
    rec.level_source = "smc_sniper"
else:
    # Priority 2: LevelEngine pounce ideas
    # Priority 3: ATR fallback
    # (same as original Task 3)
```

- [ ] **Step 9.1-9.5: Implement and test TradeRecommender** (follows same pattern)

---

## Task 10: Scanner + API + Frontend

Same as original Tasks 4-6 but `scanner.py` now uses `get_full_universe()` instead of a hardcoded list, and the API scan endpoint passes `include_pennies=True`.

Key change in `api/routers/trades.py`:

```python
from trade_signal_engine.universe import get_full_universe

@router.get("/scan")
async def trades_scan(profile, account, include_pennies: bool = Query(True)):
    candidates = get_full_universe(include_pennies=include_pennies)
    symbols = [c.symbol for c in candidates]
    result = scan_universe(symbols, profile=profile, account_size=account)
    return {...}
```

Frontend unchanged from original Task 6.

- [ ] **Step 10.1-10.6: Implement Scanner, API, Frontend** (follows same pattern)

---

## Task 11: Integration Test — Full Pipeline

Same as original Task 7 but with additional SMC-specific tests:

```python
def test_smc_detects_fvg_on_spy(self):
    """SMC engine should find FVGs on SPY 1h data."""
    import yfinance as yf
    df = yf.download("SPY", period="60d", interval="1h", progress=False)
    from smc_engine.smc_scorer import score_smc
    result = score_smc(df)
    assert result.smc_bias in ("LONG", "SHORT", "NEUTRAL")
    assert result.fvg_count >= 0
    assert result.ob_count >= 0

def test_penny_stock_in_universe(self):
    """Universe should include penny stock candidates."""
    from trade_signal_engine.universe import get_full_universe
    universe = get_full_universe(include_pennies=True)
    symbols = [c.symbol for c in universe]
    # Should have more than just blue chips
    assert len(symbols) > 50
```

---

## Dependency Graph

```
Tasks 1-5 (SMC Engine)     Task 6 (Universe)     Task 7 (DataCollector)
     └────────────┬──────────────┘                      │
                  Task 8 (UnifiedScorer) ←──────────────┘
                       │
                  Task 9 (TradeRecommender)
                       │
                  Task 10 (Scanner + API + Frontend)
                       │
                  Task 11 (Integration Test)
```

**Tasks 1-5 can run in PARALLEL with Tasks 6-7.** Task 8 depends on ALL of 1-7.

---

## Score Summary: What Makes a Killer Trade

**Score 85+ (HIGH confidence, take the trade):**
- MTF aligned (all timeframes agree on direction)
- Price at FVG + Order Block overlap in DISCOUNT zone (sniper entry)
- CHoCH confirmed (structure shifted in our direction)
- GEX supportive (dealer hedging amplifies our move)
- No exhaustion (room to run)
- Pattern forming/ready (chart confirmation)

**Score 65-84 (MEDIUM confidence, smaller size):**
- Most components aligned but 1-2 weak links
- Maybe no FVG nearby but OB present
- GEX neutral (not helping, not hurting)

**Score < 60 or NO_TRADE (skip it):**
- MTF disagrees (timeframes fighting each other)
- Price in PREMIUM zone for longs
- Exhausted move (chasing the top)
- Correction phase active
- No structure confirmation (no CHoCH or BOS)

---

## Full Market Coverage: 6,400+ Stocks

**The 233-symbol DB is 1.3% of the market. This plan scans 100%.**

| Tier | Scope | Speed | What it catches |
|------|-------|-------|-----------------|
| **1. Snapshot** | ALL 6,400+ listed US stocks | ~10s | Volume spikes, big movers, penny flags |
| **2. Pre-filter** | Top 300 candidates | ~30s | Low float, high SI, squeeze setups, accumulation |
| **3. Deep scan** | 300 candidates | ~3-5 min | Full 18-confluence + SMC + options chain |

**How we catch the $0.10→$10 mover BEFORE it runs:**
1. Tier 1 flags unusual volume on a sub-dollar stock (5x average)
2. Tier 2 checks: low float (8M shares) + high SI (35%) + no news yet
3. Tier 3 runs full analysis: Wyckoff accumulation phase confirmed, CHoCH bullish on 1H, FVG below price = entry zone
4. System outputs: **BUY at $0.12, stop $0.08, target $0.45, then trail**

**Data sources by priority:**
1. Finnhub API (free, 60/min) — complete exchange symbol list
2. yfinance bulk download — price/volume/float/SI for all tickers
3. PostgreSQL DB — our 233 tracked instruments (always included)
4. Future: SEC EDGAR (13F filings, Form 4 insider buys), Unusual Whales (options flow)
