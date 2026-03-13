"""
RoboAlgo — Vanna & Charm Exposure Engine
Computes VEX (Vanna Exposure) and CharmEX from an OptionsChain.

Why this matters:
  • GEX tracks gamma-based hedging from SPOT moves.
  • VEX tracks vanna-based hedging from IV moves → vol events trigger VEX flips.
  • Charm tracks delta-decay → heavy charm means gamma walls erode by EOD.
  • Combined: GEX + VEX → EXPLOSIVE regime; GEX alone may understate risk.

VEX formula (dealer perspective):
  call_vex = +vanna × call_OI × CONTRACT_SIZE × spot
  put_vex  = -vanna × put_OI  × CONTRACT_SIZE × spot
  net_vex  = call_vex + put_vex

CharmEX formula:
  call_charm_ex = +charm × call_OI × CONTRACT_SIZE
  put_charm_ex  = -charm × put_OI  × CONTRACT_SIZE
  net_charm_ex  = call_charm_ex + put_charm_ex
  (represents delta-units decaying per day)
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data_engine.yahoo_options_fetcher import OptionsChain, OptionContract

from options_engine.greeks_calculator import bs_vanna, bs_charm, implied_vol_approx

logger = logging.getLogger(__name__)

CONTRACT_SIZE = 100


@dataclass
class StrikeVEXRow:
    strike:        float
    call_oi:       int
    put_oi:        int
    call_vanna:    float
    put_vanna:     float
    call_vex:      float    # vanna × OI × 100 × spot
    put_vex:       float
    net_vex:       float
    call_charm:    float
    put_charm:     float
    call_charm_ex: float    # charm × OI × 100 (delta units/day)
    put_charm_ex:  float
    net_charm_ex:  float


@dataclass
class VEXProfile:
    symbol:              str
    spot:                float
    net_vex_total:       float        # $ net vanna exposure
    net_charm_ex_total:  float        # net delta units decaying per day
    vex_zero_level:      float        # strike where cumulative VEX crosses 0
    charm_decay_24h:     float        # total delta lost in next 24h from charm
    vex_regime:          str          # AMPLIFYING | DAMPENING | NEUTRAL
    vex_score:           float        # 0–100 for scanner integration
    strike_vex:          list[StrikeVEXRow] = field(default_factory=list)


class VannaExposureEngine:
    """
    Computes VEX + CharmEX profiles from an OptionsChain.

    Usage:
        engine = VannaExposureEngine()
        profile = engine.compute(chain)
    """

    def compute(self, chain: "OptionsChain", r: float = 0.045) -> VEXProfile:
        spot     = chain.spot
        contracts = chain.contracts
        if not contracts:
            return self._empty(chain.symbol, spot)

        rows = self._build_strike_vex(contracts, spot, r)
        if not rows:
            return self._empty(chain.symbol, spot)

        net_vex  = sum(r.net_vex  for r in rows)
        net_charm = sum(r.net_charm_ex for r in rows)
        charm_24h = abs(net_charm)

        # VEX zero-crossing
        vex_arr   = np.array([r.net_vex for r in rows])
        s_arr     = np.array([r.strike  for r in rows])
        order     = np.argsort(s_arr)
        cum_vex   = np.cumsum(vex_arr[order])
        s_sorted  = s_arr[order]
        zv_idx    = np.where(np.diff(np.sign(cum_vex)))[0]
        if len(zv_idx):
            i     = zv_idx[0]
            v1, v2 = cum_vex[i], cum_vex[i + 1]
            s1, s2 = s_sorted[i], s_sorted[i + 1]
            vex_zero = float(s1 + (s2 - s1) * (-v1) / (v2 - v1)) if v2 != v1 else float(s1)
        else:
            vex_zero = spot

        # Regime
        if net_vex > 0:
            vex_regime = "DAMPENING"    # VIX spike would cause dealers to buy delta (buffer)
        elif net_vex < 0:
            vex_regime = "AMPLIFYING"   # VIX spike → dealers sell delta → trend accelerates
        else:
            vex_regime = "NEUTRAL"

        # VEX score for scanner (higher = more vol-driven move potential)
        abs_net = abs(net_vex) / 1e9
        import math as _m
        mag_score = min(100, _m.log1p(abs_net) / _m.log1p(5) * 100)
        regime_score = 100 if vex_regime == "AMPLIFYING" else (50 if vex_regime == "NEUTRAL" else 20)
        vex_score = 0.6 * mag_score + 0.4 * regime_score

        return VEXProfile(
            symbol             = chain.symbol,
            spot               = spot,
            net_vex_total      = round(net_vex / 1e6, 2),     # $M
            net_charm_ex_total = round(net_charm, 0),           # delta units
            vex_zero_level     = round(vex_zero, 2),
            charm_decay_24h    = round(charm_24h, 0),
            vex_regime         = vex_regime,
            vex_score          = round(min(vex_score, 100), 1),
            strike_vex         = rows,
        )

    def _build_strike_vex(
        self,
        contracts: list["OptionContract"],
        spot: float,
        r: float,
    ) -> list[StrikeVEXRow]:
        from collections import defaultdict

        today = datetime.date.today()

        call_data: dict[float, list] = defaultdict(list)
        put_data:  dict[float, list] = defaultdict(list)

        for c in contracts:
            if c.open_interest <= 0:
                continue
            try:
                exp = datetime.date.fromisoformat(c.expiration)
                T   = max((exp - today).days / 365.0, 1 / 365.0)
            except ValueError:
                continue

            iv = c.implied_vol
            if iv <= 0.001:
                mid = (c.bid + c.ask) / 2.0 if c.ask > 0 else c.last_price
                if mid > 0.001:
                    iv = implied_vol_approx(mid, spot, c.strike, T, r, c.option_type)
            if iv <= 0.001:
                continue

            vanna  = bs_vanna(spot, c.strike, T, r, iv)
            charm  = bs_charm(spot, c.strike, T, r, iv, c.option_type)

            if c.option_type == "call":
                call_data[c.strike].append((c.open_interest, vanna, charm))
            else:
                put_data[c.strike].append((c.open_interest, vanna, charm))

        all_strikes = sorted(set(call_data) | set(put_data))
        rows: list[StrikeVEXRow] = []

        for strike in all_strikes:
            c_items = call_data.get(strike, [])
            p_items = put_data.get(strike,  [])

            c_oi    = sum(x[0] for x in c_items)
            p_oi    = sum(x[0] for x in p_items)
            c_vanna = float(np.mean([x[1] for x in c_items])) if c_items else 0.0
            p_vanna = float(np.mean([x[1] for x in p_items])) if p_items else 0.0
            c_charm = float(np.mean([x[2] for x in c_items])) if c_items else 0.0
            p_charm = float(np.mean([x[2] for x in p_items])) if p_items else 0.0

            c_vex      =  c_vanna * c_oi * CONTRACT_SIZE * spot
            p_vex      = -p_vanna * p_oi * CONTRACT_SIZE * spot
            c_charm_ex =  c_charm * c_oi * CONTRACT_SIZE
            p_charm_ex = -p_charm * p_oi * CONTRACT_SIZE

            rows.append(StrikeVEXRow(
                strike        = strike,
                call_oi       = c_oi,
                put_oi        = p_oi,
                call_vanna    = c_vanna,
                put_vanna     = p_vanna,
                call_vex      = c_vex,
                put_vex       = p_vex,
                net_vex       = c_vex + p_vex,
                call_charm    = c_charm,
                put_charm     = p_charm,
                call_charm_ex = c_charm_ex,
                put_charm_ex  = p_charm_ex,
                net_charm_ex  = c_charm_ex + p_charm_ex,
            ))
        return rows

    @staticmethod
    def _empty(symbol: str, spot: float) -> VEXProfile:
        return VEXProfile(
            symbol=symbol, spot=spot, net_vex_total=0, net_charm_ex_total=0,
            vex_zero_level=spot, charm_decay_24h=0,
            vex_regime="NEUTRAL", vex_score=0,
        )

    @staticmethod
    def to_dict(p: VEXProfile) -> dict:
        return {
            "symbol":              p.symbol,
            "spot":                p.spot,
            "net_vex_total_mn":    p.net_vex_total,
            "net_charm_ex_total":  p.net_charm_ex_total,
            "vex_zero_level":      p.vex_zero_level,
            "charm_decay_24h":     p.charm_decay_24h,
            "vex_regime":          p.vex_regime,
            "vex_score":           p.vex_score,
            "strike_vex": [
                {
                    "strike":       r.strike,
                    "net_vex":      round(r.net_vex / 1e6, 3),
                    "call_vex":     round(r.call_vex / 1e6, 3),
                    "put_vex":      round(r.put_vex / 1e6, 3),
                    "net_charm_ex": round(r.net_charm_ex, 1),
                }
                for r in p.strike_vex
            ],
        }
