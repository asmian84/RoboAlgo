"""
RoboAlgo — Market State Engine
Classifies market into four operational states that govern signal engine behavior.

States:
  COMPRESSION  → ATR_pct < 30 AND BB_width_pct < 30. Watchlist only. No trades.
  TREND        → MA20>MA50>MA200 (or inverse) + ADX rising. Trend trades allowed.
  EXPANSION    → Price broke range + ATR increasing + volume_ratio > 1.5. Breakout trades.
  CHAOS        → ATR extremely elevated + directional signals conflicting. Reduce size 50%.

Signal engine behavior per state:
  COMPRESSION  → enqueue symbol to watchlist, suppress signal output
  TREND        → allow trend-following signals, standard sizing
  EXPANSION    → allow breakout signals, can increase size up to 1.5×
  CHAOS        → allow signals but halve position sizes (50% of normal)
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, VolatilityRegime, MarketState

logger = logging.getLogger(__name__)

# ── State Thresholds ───────────────────────────────────────────────────────────
COMPRESSION_ATR_PCT   = 0.30   # ATR percentile < 30 = compressed
COMPRESSION_BB_PCT    = 0.30   # BB width percentile < 30 = compressed
CHAOS_ATR_PCT         = 0.85   # ATR percentile > 85 = chaotic
TREND_ADX_MIN         = 25.0   # ADX > 25 = trending
EXPANSION_VOL_RATIO   = 1.5    # volume_ratio > 1.5 = expansion
EXPANSION_ATR_RISE    = 0.05   # ATR rising > 5% vs prior period = expansion


class MarketStateEngine:
    """
    Classifies daily market state for all instruments.

    Usage:
        engine = MarketStateEngine()
        engine.compute_and_store()
        state = engine.get_latest("SOXL")
        size_mult = engine.get_size_multiplier("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Compute market state for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Market states"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Market state failed for {inst.symbol}: {e}")
            logger.info(f"Market state engine: stored {total} state rows.")
            return total

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the latest market state for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None
            row = session.execute(
                select(MarketState)
                .where(MarketState.instrument_id == instr.id)
                .order_by(desc(MarketState.date))
                .limit(1)
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def get_size_multiplier(self, symbol: str) -> float:
        """Return position sizing multiplier based on market state."""
        state = self.get_latest(symbol)
        if not state:
            return 1.0
        return float(state.get("size_multiplier", 1.0))

    def get_state_summary(self) -> dict:
        """Summary count of all instruments by current state."""
        with get_session() as session:
            from sqlalchemy import text
            rows = session.execute(text("""
                SELECT ms.state, COUNT(*) as cnt
                FROM market_states ms
                WHERE ms.date = (
                    SELECT MAX(date) FROM market_states ms2
                    WHERE ms2.instrument_id = ms.instrument_id
                )
                GROUP BY ms.state
                ORDER BY cnt DESC
            """)).fetchall()
            return {r[0]: r[1] for r in rows}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Compute market state rows for one instrument."""
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < 50:
            return []

        ind_rows = session.execute(
            select(Indicator)
            .where(Indicator.instrument_id == instrument.id)
            .order_by(Indicator.date)
        ).scalars().all()

        vol_rows = session.execute(
            select(VolatilityRegime)
            .where(VolatilityRegime.instrument_id == instrument.id)
            .order_by(VolatilityRegime.date)
        ).scalars().all()

        prices = pd.DataFrame([{
            "date": r.date, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume,
        } for r in price_rows]).set_index("date")

        indicators = pd.DataFrame([{
            "date": r.date, "atr": r.atr, "rsi": r.rsi,
            "ma50": r.ma50, "ma200": r.ma200,
        } for r in ind_rows]).set_index("date")

        vol_df = pd.DataFrame([{
            "date": r.date,
            "atr_percentile": r.atr_percentile,
            "bb_width_percentile": r.bb_width_percentile,
            "vol_percentile": r.vol_percentile,
        } for r in vol_rows]).set_index("date")

        df = prices.join(indicators, how="left").join(vol_df, how="left")
        df = df.dropna(subset=["close"])

        return self._classify_states(instrument.id, df)

    def _classify_states(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Classify market state for each bar."""
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"].fillna(0)
        atr    = df.get("atr", pd.Series(np.nan, index=df.index)).fillna(0)
        ma50   = df.get("ma50",  pd.Series(np.nan, index=df.index)).ffill()
        ma200  = df.get("ma200", pd.Series(np.nan, index=df.index)).ffill()

        atr_pct    = df.get("atr_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)
        bb_pct     = df.get("bb_width_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)
        vol_pct    = df.get("vol_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)

        # Derived series
        vol_avg   = volume.rolling(20).mean()
        vol_ratio = (volume / vol_avg.replace(0, np.nan)).fillna(1.0)
        atr_prev  = atr.shift(5)
        atr_change = ((atr - atr_prev) / atr_prev.replace(0, np.nan)).fillna(0)

        # MA20 (simple 20-day SMA)
        ma20 = close.rolling(20).mean()

        # ADX approximation: rolling directional movement
        adx_approx = self._compute_adx_approx(high, low, close, period=14)

        records = []
        dates = df.index.tolist()

        for i, date in enumerate(dates):
            if i < 30:
                continue

            cur_atr_pct   = float(atr_pct.iloc[i])
            cur_bb_pct    = float(bb_pct.iloc[i])
            cur_vol_pct   = float(vol_pct.iloc[i])
            cur_vol_ratio = float(vol_ratio.iloc[i])
            cur_atr_chg   = float(atr_change.iloc[i])
            cur_close     = float(close.iloc[i])
            cur_ma20      = float(ma20.iloc[i]) if not pd.isna(ma20.iloc[i]) else cur_close
            cur_ma50      = float(ma50.iloc[i]) if not pd.isna(ma50.iloc[i]) else cur_close
            cur_ma200     = float(ma200.iloc[i]) if not pd.isna(ma200.iloc[i]) else cur_close
            cur_adx       = float(adx_approx.iloc[i]) if not pd.isna(adx_approx.iloc[i]) else 20.0

            # Classify MA alignment
            if cur_ma20 > cur_ma50 and cur_ma50 > cur_ma200:
                ma_alignment = "bullish"
            elif cur_ma20 < cur_ma50 and cur_ma50 < cur_ma200:
                ma_alignment = "bearish"
            else:
                ma_alignment = "neutral"

            # State classification rules (priority order)
            state = self._classify(
                cur_atr_pct, cur_bb_pct, cur_vol_pct,
                ma_alignment, cur_adx, cur_vol_ratio, cur_atr_chg
            )

            # Compute component scores
            trend_strength     = self._trend_score(ma_alignment, cur_adx, cur_close, cur_ma50)
            expansion_strength = self._expansion_score(cur_vol_ratio, cur_atr_chg, cur_atr_pct)

            # Position sizing multiplier
            size_mult = {
                "COMPRESSION": 0.0,    # no trades
                "TREND":       1.0,
                "EXPANSION":   1.5,
                "CHAOS":       0.5,
            }.get(state, 1.0)

            records.append({
                "instrument_id":       instrument_id,
                "date":                date,
                "state":               state,
                "volatility_percentile": round(cur_vol_pct, 4),
                "trend_strength":       round(trend_strength, 2),
                "expansion_strength":   round(expansion_strength, 2),
                "adx":                  round(cur_adx, 2),
                "ma_alignment":         ma_alignment,
                "volume_ratio":         round(cur_vol_ratio, 4),
                "atr_change_pct":       round(cur_atr_chg, 4),
                "size_multiplier":      size_mult,
            })

        return records

    def _classify(
        self,
        atr_pct: float, bb_pct: float, vol_pct: float,
        ma_alignment: str, adx: float,
        vol_ratio: float, atr_change: float
    ) -> str:
        """Classify market state from indicator conditions."""
        # CHAOS: ATR extremely elevated
        if atr_pct > CHAOS_ATR_PCT and vol_ratio > 2.0:
            return "CHAOS"

        # COMPRESSION: both ATR and BB width compressed
        if atr_pct < COMPRESSION_ATR_PCT and bb_pct < COMPRESSION_BB_PCT:
            return "COMPRESSION"

        # EXPANSION: breakout conditions met
        expansion_signals = sum([
            vol_ratio >= EXPANSION_VOL_RATIO,
            atr_change >= EXPANSION_ATR_RISE,
            atr_pct > 0.60,  # ATR rising above median
        ])
        if expansion_signals >= 2:
            return "EXPANSION"

        # TREND: directional momentum
        if ma_alignment in ("bullish", "bearish") and adx >= TREND_ADX_MIN:
            return "TREND"

        # Fallback: compression (conservative default)
        return "COMPRESSION"

    def _trend_score(
        self, ma_alignment: str, adx: float, price: float, ma50: float
    ) -> float:
        """Compute 0–100 trend strength score."""
        score = 0.0
        if ma_alignment in ("bullish", "bearish"):
            score += 40.0
        score += min(adx / 50 * 40, 40.0)  # ADX contribution, max at 50
        # Price distance from MA50
        if ma50 > 0:
            dist = abs(price - ma50) / ma50
            score += min(dist / 0.1 * 20, 20.0)  # 10% above = full 20 pts
        return float(np.clip(score, 0, 100))

    def _expansion_score(
        self, vol_ratio: float, atr_change: float, atr_pct: float
    ) -> float:
        """Compute 0–100 expansion strength score."""
        vol_score   = min((vol_ratio - 1) / 2 * 40, 40.0)
        atr_score   = min(max(atr_change, 0) / 0.2 * 30, 30.0)
        pct_score   = min(atr_pct / 1.0 * 30, 30.0)
        return float(np.clip(vol_score + atr_score + pct_score, 0, 100))

    def _compute_adx_approx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Approximate ADX using True Range directional movement."""
        tr  = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        dm_plus  = ((high - high.shift(1)).clip(lower=0)
                    .where((high - high.shift(1)) > (low.shift(1) - low), 0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where((low.shift(1) - low) > (high - high.shift(1)), 0))

        atr_s    = tr.rolling(period).mean().replace(0, np.nan)
        di_plus  = (dm_plus.rolling(period).mean()  / atr_s * 100).fillna(0)
        di_minus = (dm_minus.rolling(period).mean() / atr_s * 100).fillna(0)

        dx = (((di_plus - di_minus).abs()) /
              ((di_plus + di_minus).replace(0, np.nan)) * 100).fillna(0)
        adx = dx.rolling(period).mean()
        return adx

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        stmt = pg_insert(MarketState).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_market_state_inst_date",
            set_={
                "state":                stmt.excluded.state,
                "volatility_percentile": stmt.excluded.volatility_percentile,
                "trend_strength":       stmt.excluded.trend_strength,
                "expansion_strength":   stmt.excluded.expansion_strength,
                "adx":                  stmt.excluded.adx,
                "ma_alignment":         stmt.excluded.ma_alignment,
                "volume_ratio":         stmt.excluded.volume_ratio,
                "atr_change_pct":       stmt.excluded.atr_change_pct,
                "size_multiplier":      stmt.excluded.size_multiplier,
            }
        )
        session.execute(stmt)
        session.commit()

    def _row_to_dict(self, row: MarketState) -> dict:
        return {
            "date":                  str(row.date),
            "state":                 row.state,
            "volatility_percentile": row.volatility_percentile,
            "trend_strength":        row.trend_strength,
            "expansion_strength":    row.expansion_strength,
            "adx":                   row.adx,
            "ma_alignment":          row.ma_alignment,
            "volume_ratio":          row.volume_ratio,
            "atr_change_pct":        row.atr_change_pct,
            "size_multiplier":       row.size_multiplier,
        }
