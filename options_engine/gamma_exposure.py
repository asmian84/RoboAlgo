"""
RoboAlgo — Gamma Exposure Engine
Consumes OptionsChain objects and produces the full GEX profile.

Key outputs per symbol:
  • gex_by_strike     — list of (strike, net_gex, call_gex, put_gex)
  • gex_curve         — continuous interpolated GEX curve for charting
  • call_wall         — strike with largest call OI (dealer resistance)
  • put_wall          — strike with largest put OI (dealer support)
  • zero_gamma        — price where dealer net gamma crosses zero
  • gamma_regime      — POSITIVE (dampening) | NEGATIVE (amplifying)
  • abs_total_gex     — scalar magnitude of market-wide gamma
  • gex_score         — 0–100 score for scanner integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from data_engine.yahoo_options_fetcher import OptionsChain, StrikeGEXRow

logger = logging.getLogger(__name__)

CONTRACT_SIZE = 100


@dataclass
class GEXProfile:
    symbol:         str
    spot:           float
    call_wall:      float
    put_wall:       float
    zero_gamma:     float
    gamma_regime:   str           # POSITIVE | NEGATIVE | NEUTRAL
    abs_total_gex:  float         # billions of $ exposure
    gex_score:      float         # 0–100 (higher = more useful for scanner)
    gex_flip_dist:  float         # % distance from spot to zero-gamma
    wall_spread:    float         # (call_wall − put_wall) / spot  in %
    pc_ratio:       float
    max_pain:       float
    gex_by_strike:  list[dict]    = field(default_factory=list)
    top_gex_levels: list[float]   = field(default_factory=list)
    positioning_bias: str         = "NEUTRAL"   # BULLISH | BEARISH | NEUTRAL
    # GEX sensitivity table: net GEX if spot moves to these levels
    gex_sensitivity: dict[str, float] = field(default_factory=dict)
    # DEX (Delta Exposure) by strike — net dealer delta
    dex_by_strike:  list[dict]    = field(default_factory=list)
    dex_zero_level: float         = 0.0    # strike where net dealer delta = 0
    net_dex_total:  float         = 0.0    # $ net delta exposure
    # Expanded dealer regime (5-state)
    options_regime: str           = "UNKNOWN"
    # 0DTE sub-profile summary
    gex_0dte_summary: dict        = field(default_factory=dict)


class GammaExposureEngine:
    """
    Transforms an OptionsChain into a full GEX profile.

    Usage:
        engine = GammaExposureEngine()
        profile = engine.compute(chain)
    """

    # Weights for gex_score (rocket scanner integration)
    _W_REGIME   = 35   # negative gamma gets full score (amplifies moves)
    _W_FLIP     = 30   # proximity to zero-gamma flip point
    _W_WALLS    = 20   # wall density around spot
    _W_TOTALGEX = 15   # absolute GEX magnitude (liquidity proxy)

    def compute(self, chain: OptionsChain) -> GEXProfile:
        spot = chain.spot
        rows = chain.strike_gex

        if not rows:
            return self._empty(chain.symbol, spot)

        strikes  = np.array([r.strike   for r in rows])
        net_gex  = np.array([r.net_gex  for r in rows])
        call_gex = np.array([r.call_gex for r in rows])
        put_gex  = np.array([r.put_gex  for r in rows])
        call_oi  = np.array([r.call_oi  for r in rows])
        put_oi   = np.array([r.put_oi   for r in rows])

        # ── Walls ──────────────────────────────────────────────────────────
        call_wall = float(strikes[np.argmax(call_oi)]) if call_oi.sum() > 0 else spot
        put_wall  = float(strikes[np.argmax(put_oi)])  if put_oi.sum()  > 0 else spot

        # ── Zero Gamma ─────────────────────────────────────────────────────
        zero_gamma = chain.zero_gamma or self._find_zero_gamma(strikes, net_gex, spot)

        # ── Regime ─────────────────────────────────────────────────────────
        total_net = float(np.sum(net_gex))
        regime    = "POSITIVE" if total_net > 0 else ("NEGATIVE" if total_net < 0 else "NEUTRAL")

        # ── Magnitude ──────────────────────────────────────────────────────
        abs_total = abs(total_net) / 1e9   # express in billions

        # ── Positioning bias ───────────────────────────────────────────────
        # spot above put_wall and below call_wall → neutral-to-bullish
        # spot above call_wall → bearish (dealers selling)
        # spot below put_wall  → bearish (dealers selling)
        if spot > call_wall:
            positioning_bias = "BEARISH"
        elif spot < put_wall:
            positioning_bias = "BEARISH"
        elif put_wall < spot < call_wall:
            positioning_bias = "BULLISH"
        else:
            positioning_bias = "NEUTRAL"

        # ── Scores ─────────────────────────────────────────────────────────
        # regime score: NEGATIVE γ = 100, NEUTRAL = 50, POSITIVE = 20
        regime_score = 100 if regime == "NEGATIVE" else (50 if regime == "NEUTRAL" else 20)

        # flip-distance score: closer to zero-gamma = higher instability risk
        flip_dist_pct = abs(spot - zero_gamma) / max(spot, 1e-6) * 100
        flip_score    = max(0, 100 - flip_dist_pct * 3)   # peaks at 0% distance

        # wall density: tighter walls around spot = stronger magnet
        wall_spread_pct = abs(call_wall - put_wall) / max(spot, 1e-6) * 100
        wall_score      = max(0, 100 - wall_spread_pct * 2)

        # total GEX magnitude (log-scaled 0-100)
        import math
        gex_mag_score = min(100, math.log1p(abs_total) / math.log1p(10) * 100)

        gex_score = (
            self._W_REGIME   * regime_score  / 100
            + self._W_FLIP   * flip_score    / 100
            + self._W_WALLS  * wall_score    / 100
            + self._W_TOTALGEX * gex_mag_score / 100
        )

        # ── GEX Sensitivity table ──────────────────────────────────────────
        # Recompute total net GEX as if spot moved to ±2%, ±5% scenarios
        # Uses current strike gammas scaled by (new_spot/old_spot)² (approximate)
        gex_sensitivity: dict[str, float] = {}
        for label, mult in [("dn5", 0.95), ("dn2", 0.98), ("flat", 1.0), ("up2", 1.02), ("up5", 1.05)]:
            new_spot = spot * mult
            new_gex  = sum(
                (r.call_gex - abs(r.put_gex)) * (new_spot / spot) ** 2
                for r in rows
            )
            gex_sensitivity[label] = round(new_gex / 1e9, 4)

        # ── DEX (Delta Exposure) by strike ─────────────────────────────────
        # DEX = delta × OI × CONTRACT_SIZE  (dealer perspective: short options)
        # Requires delta from contracts; approximate from chain.contracts
        dex_rows: list[dict] = []
        dex_by_strike_agg: dict[float, float] = {}
        r_free = 0.045
        for c in chain.contracts:
            import datetime
            try:
                exp  = datetime.date.fromisoformat(c.expiration)
                T    = max((exp - datetime.date.today()).days / 365.0, 1 / 365.0)
            except (ValueError, AttributeError):
                continue
            iv = c.implied_vol
            if iv <= 0.001:
                continue
            from options_engine.greeks_calculator import bs_delta as _bsd
            delta = _bsd(spot, c.strike, T, r_free, iv, c.option_type)
            # Dealer is short options → flip sign; dealer delta = -option_delta × OI
            dealer_delta = -delta * c.open_interest * CONTRACT_SIZE
            dex_by_strike_agg[c.strike] = dex_by_strike_agg.get(c.strike, 0.0) + dealer_delta

        dex_rows = [
            {"strike": k, "net_dex": round(v, 0)}
            for k, v in sorted(dex_by_strike_agg.items())
        ]
        net_dex_total = sum(dex_by_strike_agg.values())

        # DEX zero level
        dex_zero = spot
        if dex_rows:
            dex_s = np.array([d["strike"]  for d in dex_rows])
            dex_v = np.array([d["net_dex"] for d in dex_rows])
            order  = np.argsort(dex_s)
            cum_d  = np.cumsum(dex_v[order])
            s_sort = dex_s[order]
            zd_idx = np.where(np.diff(np.sign(cum_d)))[0]
            if len(zd_idx):
                i  = zd_idx[0]
                d1v, d2v = cum_d[i], cum_d[i + 1]
                s1, s2   = s_sort[i], s_sort[i + 1]
                dex_zero = float(s1 + (s2 - s1) * (-d1v) / (d2v - d1v)) if d2v != d1v else float(s1)

        # ── Expanded dealer regime (5-state) ───────────────────────────────
        if regime == "NEGATIVE" and flip_dist_pct < 3.0:
            options_regime = "EXPLOSIVE"         # short gamma + near flip → most dangerous
        elif regime == "POSITIVE" and wall_spread_pct < 3.0:
            options_regime = "PINNED"             # long gamma + tight walls → pin risk
        elif regime == "NEGATIVE":
            options_regime = "TRENDING"           # short gamma, away from flip
        elif flip_dist_pct < 5.0:
            options_regime = "TRANSITIONING"      # about to flip
        else:
            options_regime = "HEDGING"            # long gamma, standard dampening

        # ── 0DTE summary ────────────────────────────────────────────────────
        gex_0dte_summary = {}
        if chain.gex_0dte_pins:
            gex_0dte_summary = {
                "call_wall": chain.gex_0dte_call_wall,
                "put_wall":  chain.gex_0dte_put_wall,
                "top_pins":  chain.gex_0dte_pins[:3],
            }

        profile = GEXProfile(
            symbol           = chain.symbol,
            spot             = spot,
            call_wall        = call_wall,
            put_wall         = put_wall,
            zero_gamma       = round(zero_gamma, 2),
            gamma_regime     = regime,
            abs_total_gex    = round(abs_total, 4),
            gex_score        = round(min(gex_score, 100), 1),
            gex_flip_dist    = round(flip_dist_pct, 2),
            wall_spread      = round(wall_spread_pct, 2),
            pc_ratio         = chain.pc_ratio,
            max_pain         = chain.max_pain,
            positioning_bias = positioning_bias,
            gex_sensitivity  = gex_sensitivity,
            dex_by_strike    = dex_rows[:20],
            dex_zero_level   = round(dex_zero, 2),
            net_dex_total    = round(net_dex_total, 0),
            options_regime   = options_regime,
            gex_0dte_summary = gex_0dte_summary,
            gex_by_strike    = [
                {
                    "strike":   r.strike,
                    "net_gex":  round(r.net_gex / 1e6, 2),    # in $M
                    "call_gex": round(r.call_gex / 1e6, 2),
                    "put_gex":  round(r.put_gex / 1e6, 2),
                    "call_oi":  r.call_oi,
                    "put_oi":   r.put_oi,
                }
                for r in rows
            ],
            top_gex_levels   = [r["strike"] for r in chain.top_gex_strikes],
        )
        return profile

    @staticmethod
    def _find_zero_gamma(
        strikes: np.ndarray,
        net_gex: np.ndarray,
        spot: float,
    ) -> float:
        """Fallback zero-crossing finder."""
        order = np.argsort(strikes)
        s_s   = strikes[order]
        g_s   = np.cumsum(net_gex[order])
        idx   = np.where(np.diff(np.sign(g_s)))[0]
        if len(idx):
            i  = idx[0]
            g1, g2 = g_s[i], g_s[i + 1]
            s1, s2 = s_s[i], s_s[i + 1]
            return float(s1 + (s2 - s1) * (-g1) / (g2 - g1)) if g2 != g1 else float(s1)
        return spot

    @staticmethod
    def _empty(symbol: str, spot: float) -> GEXProfile:
        return GEXProfile(
            symbol=symbol, spot=spot, call_wall=0, put_wall=0,
            zero_gamma=0, gamma_regime="NEUTRAL", abs_total_gex=0,
            gex_score=0, gex_flip_dist=0, wall_spread=0, pc_ratio=0, max_pain=0,
        )

    def profile_to_dict(self, p: GEXProfile) -> dict:
        return {
            "symbol":           p.symbol,
            "spot":             p.spot,
            "call_wall":        p.call_wall,
            "put_wall":         p.put_wall,
            "zero_gamma":       p.zero_gamma,
            "gamma_regime":     p.gamma_regime,
            "abs_total_gex_bn": p.abs_total_gex,
            "gex_score":        p.gex_score,
            "gex_flip_dist_pct": p.gex_flip_dist,
            "wall_spread_pct":  p.wall_spread,
            "pc_ratio":         p.pc_ratio,
            "max_pain":         p.max_pain,
            "positioning_bias":  p.positioning_bias,
            "options_regime":    p.options_regime,
            "top_gex_levels":    p.top_gex_levels,
            "gex_by_strike":     p.gex_by_strike,
            "gex_sensitivity":   p.gex_sensitivity,
            "dex_zero_level":    p.dex_zero_level,
            "net_dex_total":     p.net_dex_total,
            "dex_by_strike":     p.dex_by_strike[:10],
            "gex_0dte_summary":  p.gex_0dte_summary,
        }
