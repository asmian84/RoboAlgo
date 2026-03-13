"""
RoboAlgo — Market Regime Timeline Engine
Aggregates market state history + trade lifecycle events into a
unified, date-indexed timeline for the Regime Timeline page.

Output shape:
  {
    "symbol":       str,
    "start_date":   str,          # ISO date
    "end_date":     str,          # ISO date
    "timeline":     [             # one row per trading day
      {
        "date":            str,   # ISO
        "market_state":    str,   # COMPRESSION/TREND/EXPANSION/CHAOS
        "trend_strength":  float | None,
        "volatility_pct":  float | None,
        "close_price":     float | None,
        "trade_event":     str | None,  # ENTRY/EXIT/SETUP/TRIGGER
        "trade_id":        int | None,
        "trade_pnl":       float | None,
        "cumulative_pnl":  float,
        "daily_pnl":       float,
      }
    ],
    "state_periods": [            # contiguous blocks of same state
      {"state": str, "start_date": str, "end_date": str, "days": int}
    ],
    "regime_stats": {             # performance by regime
      "TREND": {"trades": int, "wins": int, "total_pnl": float, "win_rate": float},
      ...
    },
    "trades": [                   # trade summary list (all EXIT trades in window)
      {"id": int, "symbol": str, "state": str, "setup_type": str,
       "entry_price": float, "exit_price": float, "pnl": float,
       "return_pct": float, "market_state": str,
       "entry_date": str, "exit_date": str}
    ],
    "generated_at": str,
  }
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── 5-minute in-memory cache ────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 300
_cache: dict = {}  # key: (symbol, start_date, end_date) → {data, expires_at}


def _cache_key(symbol: str, start: str, end: str) -> str:
    return f"{symbol}::{start}::{end}"


def _is_valid(key: str) -> bool:
    entry = _cache.get(key)
    return entry is not None and datetime.utcnow() < entry["expires_at"]


def _store(key: str, data: dict) -> None:
    _cache[key] = {
        "data":       data,
        "expires_at": datetime.utcnow() + timedelta(seconds=_CACHE_TTL_SECONDS),
    }


class RegimeTimelineEngine:
    """Build a merged market-state + trade-event timeline for a given symbol."""

    DEFAULT_LOOKBACK_DAYS = 365

    # ─────────────────────────────────────────────────────────────────────────
    def get_timeline(
        self,
        symbol:     str,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> dict:
        """
        Return the full regime timeline for *symbol* between *start_date*
        and *end_date* (both inclusive, ISO format "YYYY-MM-DD").

        Defaults:
          end_date   → today
          start_date → 365 days before end_date
        """
        # ── resolve date bounds ─────────────────────────────────────────────
        end_dt   = date.fromisoformat(end_date)   if end_date   else date.today()
        start_dt = date.fromisoformat(start_date) if start_date else end_dt - timedelta(days=self.DEFAULT_LOOKBACK_DAYS)

        start_str = start_dt.isoformat()
        end_str   = end_dt.isoformat()

        key = _cache_key(symbol, start_str, end_str)
        if _is_valid(key):
            return _cache[key]["data"]

        result = self._build_timeline(symbol, start_dt, end_dt)
        _store(key, result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _build_timeline(self, symbol: str, start_dt: date, end_dt: date) -> dict:
        from database.connection import get_session
        from database.models import Instrument, MarketState, TradeLifecycle, PriceData
        from sqlalchemy import select, and_, desc

        # ── 1. Fetch instrument id ──────────────────────────────────────────
        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == symbol)
            ).scalar_one_or_none()

            if inst is None:
                return self._empty(symbol, start_dt.isoformat(), end_dt.isoformat(),
                                   error=f"Symbol {symbol} not found")

            inst_id = inst.id

            # ── 2. Market states in window ──────────────────────────────────
            ms_rows = session.execute(
                select(MarketState)
                .where(
                    and_(
                        MarketState.instrument_id == inst_id,
                        MarketState.date >= start_dt,
                        MarketState.date <= end_dt,
                    )
                )
                .order_by(MarketState.date)
            ).scalars().all()

            # ── 3. Price data (close) in window ────────────────────────────
            price_rows = session.execute(
                select(PriceData)
                .where(
                    and_(
                        PriceData.instrument_id == inst_id,
                        PriceData.date >= start_dt,
                        PriceData.date <= end_dt,
                    )
                )
                .order_by(PriceData.date)
            ).scalars().all()

            # ── 4. Trades touching this window ─────────────────────────────
            trade_rows = session.execute(
                select(TradeLifecycle)
                .where(
                    and_(
                        TradeLifecycle.symbol == symbol,
                        TradeLifecycle.state == "EXIT",
                        TradeLifecycle.entry_timestamp >= datetime.combine(start_dt, datetime.min.time()),
                        TradeLifecycle.exit_timestamp  <= datetime.combine(end_dt,   datetime.max.time()),
                    )
                )
                .order_by(TradeLifecycle.entry_timestamp)
            ).scalars().all()

        # ── Build lookup dicts ──────────────────────────────────────────────
        state_by_date:  dict[str, MarketState]  = {str(r.date): r for r in ms_rows}
        price_by_date:  dict[str, float]        = {
            str(r.date): r.close for r in price_rows if r.close is not None
        }

        # Build trade event maps: entry_date → list[trade], exit_date → list[trade]
        entry_events: dict[str, list] = {}
        exit_events:  dict[str, list] = {}
        for t in trade_rows:
            if t.entry_timestamp:
                ed = str(t.entry_timestamp.date())
                entry_events.setdefault(ed, []).append(t)
            if t.exit_timestamp:
                xd = str(t.exit_timestamp.date())
                exit_events.setdefault(xd, []).append(t)

        # ── Build daily timeline ───────────────────────────────────────────
        cumulative_pnl = 0.0
        timeline       = []
        current        = start_dt

        while current <= end_dt:
            ds = current.isoformat()
            ms = state_by_date.get(ds)

            # Sum P&L for exits on this date
            daily_pnl  = sum(t.pnl or 0.0 for t in exit_events.get(ds, []))
            cumulative_pnl += daily_pnl

            # Trade event label (prefer EXIT over ENTRY when both occur same day)
            trade_event = None
            trade_id    = None
            trade_pnl   = None

            if ds in exit_events:
                t = exit_events[ds][0]
                trade_event = "EXIT"
                trade_id    = t.id
                trade_pnl   = t.pnl
            elif ds in entry_events:
                t = entry_events[ds][0]
                trade_event = "ENTRY"
                trade_id    = t.id
                trade_pnl   = None

            timeline.append({
                "date":            ds,
                "market_state":    ms.state if ms else "UNKNOWN",
                "trend_strength":  round(ms.trend_strength, 2) if ms and ms.trend_strength is not None else None,
                "volatility_pct":  round(ms.volatility_percentile, 3) if ms and ms.volatility_percentile is not None else None,
                "close_price":     price_by_date.get(ds),
                "trade_event":     trade_event,
                "trade_id":        trade_id,
                "trade_pnl":       trade_pnl,
                "cumulative_pnl":  round(cumulative_pnl, 2),
                "daily_pnl":       round(daily_pnl, 2),
            })

            current += timedelta(days=1)

        # ── State period blocks ────────────────────────────────────────────
        state_periods = self._build_state_periods(timeline)

        # ── Regime stats ───────────────────────────────────────────────────
        regime_stats = self._build_regime_stats(trade_rows, state_by_date)

        # ── Trade summary list ─────────────────────────────────────────────
        trades_summary = []
        for t in trade_rows:
            trades_summary.append({
                "id":          t.id,
                "symbol":      t.symbol,
                "state":       t.state,
                "setup_type":  t.setup_type,
                "market_state": t.market_state,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "pnl":         round(t.pnl, 2) if t.pnl is not None else None,
                "return_pct":  round(t.return_percent, 2) if t.return_percent is not None else None,
                "entry_date":  str(t.entry_timestamp.date()) if t.entry_timestamp else None,
                "exit_date":   str(t.exit_timestamp.date())  if t.exit_timestamp  else None,
                "holding_days": t.holding_period,
                "exit_reason": t.exit_reason,
            })

        return {
            "symbol":        symbol,
            "start_date":    start_dt.isoformat(),
            "end_date":      end_dt.isoformat(),
            "timeline":      timeline,
            "state_periods": state_periods,
            "regime_stats":  regime_stats,
            "trades":        trades_summary,
            "generated_at":  datetime.utcnow().isoformat() + "Z",
        }

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_state_periods(timeline: list) -> list:
        """Collapse consecutive same-state days into period blocks."""
        if not timeline:
            return []

        periods = []
        current_state = timeline[0]["market_state"]
        period_start  = timeline[0]["date"]
        period_end    = timeline[0]["date"]

        for row in timeline[1:]:
            if row["market_state"] == current_state:
                period_end = row["date"]
            else:
                periods.append({
                    "state":      current_state,
                    "start_date": period_start,
                    "end_date":   period_end,
                    "days":       (
                        date.fromisoformat(period_end) -
                        date.fromisoformat(period_start)
                    ).days + 1,
                })
                current_state = row["market_state"]
                period_start  = row["date"]
                period_end    = row["date"]

        # Flush last period
        periods.append({
            "state":      current_state,
            "start_date": period_start,
            "end_date":   period_end,
            "days":       (
                date.fromisoformat(period_end) -
                date.fromisoformat(period_start)
            ).days + 1,
        })

        return periods

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_regime_stats(trade_rows: list, state_by_date: dict) -> dict:
        """
        Aggregate completed trades by the market state at entry date.
        Returns win rate and total P&L per regime.
        """
        stats: dict[str, dict] = {}

        for t in trade_rows:
            if t.entry_timestamp is None:
                continue

            entry_date  = str(t.entry_timestamp.date())
            ms_row      = state_by_date.get(entry_date)
            regime      = ms_row.state if ms_row else (t.market_state or "UNKNOWN")

            if regime not in stats:
                stats[regime] = {
                    "trades":    0,
                    "wins":      0,
                    "total_pnl": 0.0,
                    "win_rate":  0.0,
                    "avg_pnl":   0.0,
                }

            stats[regime]["trades"] += 1
            pnl = t.pnl or 0.0
            stats[regime]["total_pnl"] += pnl
            if pnl > 0:
                stats[regime]["wins"] += 1

        # Finalize rates
        for regime, s in stats.items():
            if s["trades"] > 0:
                s["win_rate"] = round(s["wins"] / s["trades"], 3)
                s["avg_pnl"]  = round(s["total_pnl"] / s["trades"], 2)
            s["total_pnl"] = round(s["total_pnl"], 2)

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _empty(symbol: str, start: str, end: str, error: str = "") -> dict:
        return {
            "symbol":        symbol,
            "start_date":    start,
            "end_date":      end,
            "timeline":      [],
            "state_periods": [],
            "regime_stats":  {},
            "trades":        [],
            "generated_at":  datetime.utcnow().isoformat() + "Z",
            "error":         error,
        }
