"""Capital Flow Engine.

Detects cross-sector capital flow using momentum, volume expansion, volatility
expansion, and breakout participation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re

import pandas as pd
from sqlalchemy import text

from config.settings import (
    COMMODITY_LEVERAGED,
    CRYPTO_ETFS,
    INDEX_DRIVERS,
    INDEX_LEVERAGED,
    SECTOR_LEVERAGED,
    SINGLE_STOCK_LEVERAGED,
)
from database.connection import get_engine, get_session

# Weights requested in spec
W_MOMENTUM = 0.40
W_VOLUME_EXPANSION = 0.25
W_VOLATILITY_EXPANSION = 0.20
W_BREAKOUT_COUNT = 0.15


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: float | int | None, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return default
        return fv
    except Exception:
        return default


class CapitalFlowEngine:
    """Computes sector-level capital flow and stores it in sector_flow_scores."""

    def __init__(self) -> None:
        self._sector_by_symbol = self._build_sector_map()

    def compute_and_store(self, max_symbols: int = 5000, lookback_bars: int = 90) -> list[dict]:
        self._ensure_table()
        prices = self._load_prices(max_symbols=max_symbols, lookback_bars=lookback_bars)
        if prices.empty:
            return []

        per_symbol = self._compute_symbol_features(prices)
        if per_symbol.empty:
            return []

        per_symbol["sector"] = per_symbol.apply(
            lambda r: self._infer_sector(
                symbol=str(r["symbol"]),
                name=str(r.get("name", "") or ""),
                underlying=str(r.get("underlying", "") or ""),
            ),
            axis=1,
        )

        breakout_map = self._load_breakout_flags(per_symbol["symbol"].tolist())
        per_symbol["has_breakout"] = per_symbol["symbol"].map(lambda s: breakout_map.get(str(s), 0))

        result = self._score_sectors(per_symbol)
        self._persist(result)
        return result

    def get_latest(self, limit: int = 20) -> list[dict]:
        q = text(
            """
            SELECT sector, flow_score, top_symbols, computed_at
            FROM sector_flow_scores
            ORDER BY score_date DESC, flow_score DESC
            LIMIT :limit
            """
        )
        with get_session() as session:
            rows = session.execute(q, {"limit": limit}).mappings().all()
        return [
            {
                "sector": row["sector"],
                "flow_score": _safe_float(row["flow_score"]),
                "top_symbols": row["top_symbols"] or [],
                "computed_at": row["computed_at"].isoformat() if row["computed_at"] else None,
            }
            for row in rows
        ]

    def _ensure_table(self) -> None:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS sector_flow_scores (
                      id BIGSERIAL PRIMARY KEY,
                      sector VARCHAR(80) NOT NULL,
                      score_date DATE NOT NULL,
                      flow_score DOUBLE PRECISION NOT NULL,
                      top_symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
                      momentum_score DOUBLE PRECISION,
                      volume_expansion_score DOUBLE PRECISION,
                      volatility_expansion_score DOUBLE PRECISION,
                      breakout_count_score DOUBLE PRECISION,
                      computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                      UNIQUE (sector, score_date)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_sector_flow_date_score
                    ON sector_flow_scores (score_date DESC, flow_score DESC)
                    """
                )
            )

    def _load_prices(self, max_symbols: int, lookback_bars: int) -> pd.DataFrame:
        q = text(
            """
            WITH selected AS (
              SELECT id, symbol, COALESCE(name, '') AS name, COALESCE(underlying, '') AS underlying
              FROM instruments
              ORDER BY symbol
              LIMIT :max_symbols
            ),
            ranked AS (
              SELECT
                p.instrument_id,
                p.date,
                p.close,
                p.volume,
                ROW_NUMBER() OVER (PARTITION BY p.instrument_id ORDER BY p.date DESC) AS rn
              FROM price_data p
              JOIN selected s ON s.id = p.instrument_id
              WHERE p.close IS NOT NULL
            )
            SELECT s.symbol, s.name, s.underlying, r.date, r.close, r.volume
            FROM ranked r
            JOIN selected s ON s.id = r.instrument_id
            WHERE r.rn <= :lookback_bars
            ORDER BY s.symbol, r.date
            """
        )
        return pd.read_sql_query(
            q,
            get_engine(),
            params={"max_symbols": max_symbols, "lookback_bars": lookback_bars},
        )

    def _compute_symbol_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for symbol, g in prices.groupby("symbol", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            if len(g) < 25:
                continue

            close = g["close"].astype(float)
            vol = g["volume"].fillna(0.0).astype(float)

            lookback = min(20, len(close) - 1)
            if lookback < 5:
                continue
            momentum = _safe_float((close.iloc[-1] / close.iloc[-1 - lookback]) - 1.0)

            v_recent = _safe_float(vol.tail(5).mean(), default=0.0)
            v_prev = _safe_float(vol.iloc[max(0, len(vol) - 25) : max(0, len(vol) - 5)].mean(), default=0.0)
            volume_expansion = (v_recent / v_prev) if v_prev > 0 else 1.0

            rets = close.pct_change().dropna()
            if len(rets) >= 60:
                vol_20 = _safe_float(rets.tail(20).std())
                vol_60 = _safe_float(rets.tail(60).std())
            elif len(rets) >= 20:
                vol_20 = _safe_float(rets.tail(20).std())
                vol_60 = _safe_float(rets.std())
            else:
                vol_20 = 0.0
                vol_60 = 0.0
            volatility_expansion = (vol_20 / vol_60) if vol_60 > 0 else 1.0

            meta = g.iloc[-1]
            rows.append(
                {
                    "symbol": symbol,
                    "name": str(meta.get("name", "") or ""),
                    "underlying": str(meta.get("underlying", "") or ""),
                    "momentum": momentum,
                    "volume_expansion": _safe_float(volume_expansion, 1.0),
                    "volatility_expansion": _safe_float(volatility_expansion, 1.0),
                }
            )
        return pd.DataFrame(rows)

    def _load_breakout_flags(self, symbols: list[str]) -> dict[str, int]:
        if not symbols:
            return {}

        q = text(
            """
            WITH latest AS (
              SELECT
                i.symbol,
                b.breakout_strength,
                b.date,
                ROW_NUMBER() OVER (PARTITION BY i.symbol ORDER BY b.date DESC) AS rn
              FROM breakout_signals b
              JOIN instruments i ON i.id = b.instrument_id
              WHERE i.symbol = ANY(:symbols)
            )
            SELECT symbol,
                   CASE WHEN breakout_strength >= 60 THEN 1 ELSE 0 END AS has_breakout
            FROM latest
            WHERE rn = 1
            """
        )
        with get_session() as session:
            rows = session.execute(q, {"symbols": symbols}).mappings().all()
        return {str(r["symbol"]): int(r["has_breakout"]) for r in rows}

    def _score_sectors(self, per_symbol: pd.DataFrame) -> list[dict]:
        out: list[dict] = []
        grouped = per_symbol.groupby("sector", sort=False)

        for sector, g in grouped:
            if g.empty:
                continue

            momentum_raw = _safe_float(g["momentum"].mean())
            volume_raw = _safe_float(g["volume_expansion"].mean(), 1.0)
            volx_raw = _safe_float(g["volatility_expansion"].mean(), 1.0)
            breakout_rate = _safe_float(g["has_breakout"].mean())

            # Normalize each factor to 0-100 for weighted scoring
            momentum_score = _clamp(50.0 + momentum_raw * 500.0, 0.0, 100.0)
            volume_score = _clamp((volume_raw - 0.5) / 1.5 * 100.0, 0.0, 100.0)
            volx_score = _clamp((volx_raw - 0.6) / 1.0 * 100.0, 0.0, 100.0)
            breakout_score = _clamp(breakout_rate * 100.0, 0.0, 100.0)

            flow_score = (
                W_MOMENTUM * momentum_score
                + W_VOLUME_EXPANSION * volume_score
                + W_VOLATILITY_EXPANSION * volx_score
                + W_BREAKOUT_COUNT * breakout_score
            )

            top = (
                g.assign(_rank=g["momentum"] * 0.6 + g["volume_expansion"] * 0.25 + g["has_breakout"] * 0.15)
                .sort_values("_rank", ascending=False)["symbol"]
                .head(8)
                .tolist()
            )

            out.append(
                {
                    "sector": sector,
                    "flow_score": round(_safe_float(flow_score), 2),
                    "top_symbols": top,
                    "momentum_score": round(momentum_score, 2),
                    "volume_expansion_score": round(volume_score, 2),
                    "volatility_expansion_score": round(volx_score, 2),
                    "breakout_count_score": round(breakout_score, 2),
                }
            )

        out.sort(key=lambda x: x["flow_score"], reverse=True)
        return out

    def _persist(self, rows: list[dict]) -> None:
        if not rows:
            return
        upsert = text(
            """
            INSERT INTO sector_flow_scores (
              sector, score_date, flow_score, top_symbols,
              momentum_score, volume_expansion_score, volatility_expansion_score,
              breakout_count_score, computed_at
            )
            VALUES (
              :sector, CURRENT_DATE, :flow_score, CAST(:top_symbols AS JSONB),
              :momentum_score, :volume_expansion_score, :volatility_expansion_score,
              :breakout_count_score, :computed_at
            )
            ON CONFLICT (sector, score_date)
            DO UPDATE SET
              flow_score = EXCLUDED.flow_score,
              top_symbols = EXCLUDED.top_symbols,
              momentum_score = EXCLUDED.momentum_score,
              volume_expansion_score = EXCLUDED.volume_expansion_score,
              volatility_expansion_score = EXCLUDED.volatility_expansion_score,
              breakout_count_score = EXCLUDED.breakout_count_score,
              computed_at = EXCLUDED.computed_at
            """
        )
        now = datetime.now(timezone.utc)
        payload = [
            {
                "sector": r["sector"],
                "flow_score": r["flow_score"],
                "top_symbols": json.dumps(r["top_symbols"]),
                "momentum_score": r["momentum_score"],
                "volume_expansion_score": r["volume_expansion_score"],
                "volatility_expansion_score": r["volatility_expansion_score"],
                "breakout_count_score": r["breakout_count_score"],
                "computed_at": now,
            }
            for r in rows
        ]
        with get_session() as session:
            session.execute(upsert, payload)
            session.commit()

    def _build_sector_map(self) -> dict[str, str]:
        sector_by_symbol: dict[str, str] = {}

        def add_pair_list(pairs: list[tuple], sector_label: str | None = None) -> None:
            for bull, bear, desc, *_ in pairs:
                label = sector_label or self._sector_from_text(str(desc))
                sector_by_symbol[bull] = label
                if bear:
                    sector_by_symbol[bear] = label

        add_pair_list(SECTOR_LEVERAGED)
        add_pair_list(INDEX_LEVERAGED, "Broad Market")
        add_pair_list(COMMODITY_LEVERAGED, "Commodities")
        add_pair_list(SINGLE_STOCK_LEVERAGED, "Single Stock")

        for s in INDEX_DRIVERS:
            sector_by_symbol[s] = "Broad Market"
        for s in CRYPTO_ETFS:
            sector_by_symbol[s] = "Crypto"

        return sector_by_symbol

    def _infer_sector(self, symbol: str, name: str, underlying: str) -> str:
        symbol_u = symbol.upper()
        if symbol_u in self._sector_by_symbol:
            return self._sector_by_symbol[symbol_u]

        text_blob = f"{name} {underlying} {symbol_u}".lower()
        return self._sector_from_text(text_blob)

    @staticmethod
    def _sector_from_text(text_blob: str) -> str:
        t = re.sub(r"\s+", " ", text_blob.lower()).strip()
        if any(k in t for k in ("semiconductor", "soxx", "chip", "nvda", "amd", "tsm", "smci")):
            return "Semiconductors"
        if any(k in t for k in ("financial", "bank", "xlf", "jpm", "goldman", "ms ", "wfc", "bac", "citi")):
            return "Financials"
        if any(k in t for k in ("energy", "oil", "gas", "xle", "xom", "cvx", "cop")):
            return "Energy"
        if any(k in t for k in ("health", "biotech", "pharma", "xlv", "lly", "jnj", "abbv", "amgn")):
            return "Healthcare"
        if any(k in t for k in ("tech", "technology", "software", "cloud", "xlk", "qqq", "aapl", "msft", "meta", "googl", "amzn")):
            return "Tech"
        if any(k in t for k in ("crypto", "bitcoin", "ethereum", "coin", "mstr", "mara", "ibit", "fbtc", "etha", "xrpi")):
            return "Crypto"
        if any(k in t for k in ("retail", "consumer", "xrt", "xly", "web", "homebuilder", "itb")):
            return "Consumer"
        if any(k in t for k in ("gold", "silver", "crude", "natural gas", "commodity")):
            return "Commodities"
        return "Broad Market"
