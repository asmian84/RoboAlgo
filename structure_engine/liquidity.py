"""
RoboAlgo — Liquidity Engine
Identifies key price levels where institutional liquidity pools exist.

Key levels tracked:
  - Previous day high/low
  - 3-day high/low
  - 5-day (weekly) high/low
  - VWAP approximation
  - Volume-profile high-volume nodes

Sweep events: price crosses these levels and reverses (liquidity grab).
Breakouts through liquidity levels increase confluence score.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, LiquidityLevel

logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
LOOKBACK_3D   = 3
LOOKBACK_WEEK = 5   # trading days in a week
VWAP_WINDOW   = 20  # bars for VWAP approximation
VOL_NODE_BINS = 20  # number of price bins for volume profile
SWEEP_MARGIN  = 0.003  # 0.3% — price must exceed level by this to count as sweep


class LiquidityEngine:
    """
    Computes and stores key liquidity levels with sweep events.

    Usage:
        engine = LiquidityEngine()
        engine.compute_and_store()
        levels = engine.get_latest("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Compute liquidity levels for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Liquidity levels"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Liquidity engine failed for {inst.symbol}: {e}")
            logger.info(f"Liquidity engine: stored {total} level rows.")
            return total

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the latest liquidity level data for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None
            row = session.execute(
                select(LiquidityLevel)
                .where(LiquidityLevel.instrument_id == instr.id)
                .order_by(desc(LiquidityLevel.date))
                .limit(1)
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def score_for_breakout(self, symbol: str, breakout_direction: str = "up") -> float:
        """
        Return a 0–100 liquidity score for a breakout in a given direction.
        Higher = more liquidity levels cleared, stronger breakout validation.
        """
        latest = self.get_latest(symbol)
        if not latest:
            return 50.0

        score = 50.0  # base score

        # Swept levels increase conviction
        if breakout_direction == "up":
            if latest.get("swept_prev_high"):
                score += 15
            if latest.get("swept_week_high"):
                score += 10
        else:
            if latest.get("swept_prev_low"):
                score += 15
            if latest.get("swept_week_low"):
                score += 10

        # VWAP position
        if breakout_direction == "up" and latest.get("above_vwap"):
            score += 10
        elif breakout_direction == "down" and not latest.get("above_vwap"):
            score += 10

        # Near high-volume node (support/resistance cluster)
        if latest.get("near_vol_node"):
            score += 5

        return float(np.clip(score, 0, 100))

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Compute liquidity levels across the full price history."""
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < LOOKBACK_WEEK + 5:
            return []

        prices = pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in price_rows])
        prices = prices.set_index("date")

        return self._compute_levels(instrument.id, prices)

    def _compute_levels(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Compute all liquidity level metrics for each bar."""
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"].fillna(0)

        # Rolling reference levels
        prev_high = high.shift(1)
        prev_low  = low.shift(1)
        high_3d   = high.rolling(LOOKBACK_3D).max().shift(1)
        low_3d    = low.rolling(LOOKBACK_3D).min().shift(1)
        high_week = high.rolling(LOOKBACK_WEEK).max().shift(1)
        low_week  = low.rolling(LOOKBACK_WEEK).min().shift(1)

        # VWAP approximation: cumulative(typical_price × volume) / cumulative(volume)
        # Rolling 20-bar VWAP
        typical = (high + low + close) / 3
        vwap = (
            (typical * volume).rolling(VWAP_WINDOW).sum()
            / volume.rolling(VWAP_WINDOW).sum().replace(0, np.nan)
        )

        # Volume profile: rolling high-volume price nodes
        vol_node_high, vol_node_low = self._compute_volume_nodes(df)

        records = []
        dates = df.index.tolist()

        for i, date in enumerate(dates):
            if i < LOOKBACK_WEEK + 1:
                continue

            cur_high  = float(high.iloc[i])
            cur_low   = float(low.iloc[i])
            cur_close = float(close.iloc[i])
            p_high    = float(prev_high.iloc[i]) if not pd.isna(prev_high.iloc[i]) else cur_high
            p_low     = float(prev_low.iloc[i])  if not pd.isna(prev_low.iloc[i])  else cur_low
            h3        = float(high_3d.iloc[i])   if not pd.isna(high_3d.iloc[i])   else cur_high
            l3        = float(low_3d.iloc[i])    if not pd.isna(low_3d.iloc[i])    else cur_low
            h_wk      = float(high_week.iloc[i]) if not pd.isna(high_week.iloc[i]) else cur_high
            l_wk      = float(low_week.iloc[i])  if not pd.isna(low_week.iloc[i])  else cur_low
            vwap_val  = float(vwap.iloc[i])      if not pd.isna(vwap.iloc[i])      else cur_close
            vn_high   = float(vol_node_high.iloc[i]) if not pd.isna(vol_node_high.iloc[i]) else cur_high
            vn_low    = float(vol_node_low.iloc[i])  if not pd.isna(vol_node_low.iloc[i])  else cur_low

            # Sweep detection: bar exceeded level intrabar then moved away
            swept_prev_high  = (cur_high > p_high * (1 + SWEEP_MARGIN)) and (cur_close < p_high)
            swept_prev_low   = (cur_low  < p_low  * (1 - SWEEP_MARGIN)) and (cur_close > p_low)
            swept_week_high  = (cur_high > h_wk   * (1 + SWEEP_MARGIN)) and (cur_close < h_wk)
            swept_week_low   = (cur_low  < l_wk   * (1 - SWEEP_MARGIN)) and (cur_close > l_wk)

            above_vwap       = cur_close > vwap_val
            near_vol_node    = (
                abs(cur_close - vn_high) / max(cur_close, 1e-8) < 0.01 or
                abs(cur_close - vn_low)  / max(cur_close, 1e-8) < 0.01
            )

            # Liquidity score
            score = self._compute_liquidity_score(
                cur_close, p_high, p_low, h_wk, l_wk, vwap_val,
                swept_prev_high, swept_prev_low, above_vwap, near_vol_node
            )

            records.append({
                "instrument_id":   instrument_id,
                "date":            date,
                "prev_day_high":   round(p_high, 4),
                "prev_day_low":    round(p_low, 4),
                "high_3d":         round(h3, 4),
                "low_3d":          round(l3, 4),
                "high_week":       round(h_wk, 4),
                "low_week":        round(l_wk, 4),
                "vwap":            round(vwap_val, 4),
                "vol_node_high":   round(vn_high, 4),
                "vol_node_low":    round(vn_low, 4),
                "swept_prev_high": swept_prev_high,
                "swept_prev_low":  swept_prev_low,
                "swept_week_high": swept_week_high,
                "swept_week_low":  swept_week_low,
                "above_vwap":      above_vwap,
                "near_vol_node":   near_vol_node,
                "liquidity_score": round(score, 2),
            })

        return records

    def _compute_volume_nodes(
        self, df: pd.DataFrame, lookback: int = 30
    ) -> tuple[pd.Series, pd.Series]:
        """
        Identify high-volume price zones via rolling volume profile.
        Returns (high_node, low_node) as price Series.
        """
        high   = df["high"]
        low    = df["low"]
        close  = df["close"]
        volume = df["volume"].fillna(0)

        high_nodes = pd.Series(np.nan, index=df.index)
        low_nodes  = pd.Series(np.nan, index=df.index)

        for i in range(lookback, len(df)):
            slice_high   = high.iloc[i-lookback:i].values
            slice_low    = low.iloc[i-lookback:i].values
            slice_vol    = volume.iloc[i-lookback:i].values
            slice_close  = close.iloc[i-lookback:i].values

            price_min = slice_low.min()
            price_max = slice_high.max()

            if price_max <= price_min:
                continue

            # Create price bins and accumulate volume
            bins = np.linspace(price_min, price_max, VOL_NODE_BINS + 1)
            vol_by_bin = np.zeros(VOL_NODE_BINS)
            for j in range(len(slice_close)):
                bin_idx = min(int((slice_close[j] - price_min)
                                  / (price_max - price_min) * VOL_NODE_BINS),
                              VOL_NODE_BINS - 1)
                vol_by_bin[bin_idx] += slice_vol[j]

            # Highest volume bin
            top_bin = np.argmax(vol_by_bin)
            node_price = (bins[top_bin] + bins[top_bin + 1]) / 2

            high_nodes.iloc[i] = node_price if node_price > slice_close[-1] else price_max
            low_nodes.iloc[i]  = node_price if node_price < slice_close[-1] else price_min

        return high_nodes, low_nodes

    def _compute_liquidity_score(
        self, price: float, prev_high: float, prev_low: float,
        week_high: float, week_low: float, vwap: float,
        swept_prev_high: bool, swept_prev_low: bool,
        above_vwap: bool, near_vol_node: bool
    ) -> float:
        """Compute 0–100 liquidity confluence score."""
        score = 50.0  # neutral base

        # Proximity to key levels increases awareness
        for level in [prev_high, prev_low, week_high, week_low]:
            if level and abs(price - level) / max(price, 1e-8) < 0.005:  # within 0.5%
                score += 5

        # Sweeps boost score (institutional activity detected)
        if swept_prev_high or swept_prev_low:
            score += 10
        if above_vwap:
            score += 5
        if near_vol_node:
            score += 5

        return float(np.clip(score, 0, 100))

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        stmt = pg_insert(LiquidityLevel).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_liq_inst_date",
            set_={
                "prev_day_high":  stmt.excluded.prev_day_high,
                "prev_day_low":   stmt.excluded.prev_day_low,
                "high_3d":        stmt.excluded.high_3d,
                "low_3d":         stmt.excluded.low_3d,
                "high_week":      stmt.excluded.high_week,
                "low_week":       stmt.excluded.low_week,
                "vwap":           stmt.excluded.vwap,
                "vol_node_high":  stmt.excluded.vol_node_high,
                "vol_node_low":   stmt.excluded.vol_node_low,
                "swept_prev_high": stmt.excluded.swept_prev_high,
                "swept_prev_low":  stmt.excluded.swept_prev_low,
                "swept_week_high": stmt.excluded.swept_week_high,
                "swept_week_low":  stmt.excluded.swept_week_low,
                "above_vwap":     stmt.excluded.above_vwap,
                "near_vol_node":  stmt.excluded.near_vol_node,
                "liquidity_score": stmt.excluded.liquidity_score,
            }
        )
        session.execute(stmt)
        session.commit()

    def _row_to_dict(self, row: LiquidityLevel) -> dict:
        return {
            "date":            str(row.date),
            "prev_day_high":   row.prev_day_high,
            "prev_day_low":    row.prev_day_low,
            "high_3d":         row.high_3d,
            "low_3d":          row.low_3d,
            "high_week":       row.high_week,
            "low_week":        row.low_week,
            "vwap":            row.vwap,
            "vol_node_high":   row.vol_node_high,
            "vol_node_low":    row.vol_node_low,
            "swept_prev_high": row.swept_prev_high,
            "swept_prev_low":  row.swept_prev_low,
            "swept_week_high": row.swept_week_high,
            "swept_week_low":  row.swept_week_low,
            "above_vwap":      row.above_vwap,
            "near_vol_node":   row.near_vol_node,
            "liquidity_score": row.liquidity_score,
        }
