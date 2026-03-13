"""
RoboAlgo — Black-Scholes Greeks Calculator
Computes delta, gamma, theta, vega, rho analytically.
Used when yfinance does not supply greeks (e.g., deep OTM or illiquid contracts).

All inputs:
    S   — current spot price
    K   — strike price
    T   — time to expiry in years
    r   — risk-free rate (decimal, e.g. 0.05)
    sigma — implied volatility (annualised decimal)
    q   — continuous dividend yield (default 0)
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np


_SQRT_2PI = math.sqrt(2 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, sigma: float, T: float) -> float:
    return d1_val - sigma * math.sqrt(T)


def _N(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


# ─────────────────────────────────────────────────────────────────────────────

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(d1v, sigma, T)
    if option_type == "call":
        return S * math.exp(-q * T) * _N(d1v) - K * math.exp(-r * T) * _N(d2v)
    return K * math.exp(-r * T) * _N(-d2v) - S * math.exp(-q * T) * _N(-d1v)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    if T <= 0 or sigma <= 0:
        return (1.0 if S >= K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
    d1v = _d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return math.exp(-q * T) * _N(d1v)
    return math.exp(-q * T) * (_N(d1v) - 1)


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """Gamma is identical for calls and puts."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1v = _d1(S, K, T, r, sigma, q)
    return math.exp(-q * T) * _n(d1v) / (S * sigma * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    annualised: bool = False,
) -> float:
    """Theta per calendar day (or annualised if annualised=True)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(d1v, sigma, T)
    nd1 = _n(d1v)
    t1  = -(S * math.exp(-q * T) * nd1 * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        t2  = r * K * math.exp(-r * T) * _N(d2v)
        t3  = q * S * math.exp(-q * T) * _N(d1v)
        theta_annual = t1 - t2 + t3
    else:
        t2  = r * K * math.exp(-r * T) * _N(-d2v)
        t3  = q * S * math.exp(-q * T) * _N(-d1v)
        theta_annual = t1 + t2 - t3
    return theta_annual if annualised else theta_annual / 365.0


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """Vega per 1% move in IV (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1v = _d1(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _n(d1v) * math.sqrt(T) * 0.01


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    """Rho per 1% move in rates."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(d1v, sigma, T)
    if option_type == "call":
        return K * T * math.exp(-r * T) * _N(d2v)  * 0.01
    return -K * T * math.exp(-r * T) * _N(-d2v) * 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Second-order & cross Greeks — Vanna, Charm, DEX
# ─────────────────────────────────────────────────────────────────────────────

def bs_vanna(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """
    Vanna = ∂²V / ∂S∂σ = ∂delta/∂σ = ∂vega/∂S
    Measures how delta changes as implied volatility moves (same for calls/puts).
    Key for VEX: large |vanna| × OI means big delta-hedging as vol changes.

    Formula: vanna = -exp(-q·T) · N_pdf(d1) · d2/σ
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(d1v, sigma, T)
    return -math.exp(-q * T) * _n(d1v) * d2v / sigma


def bs_charm(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    """
    Charm = ∂delta/∂t (delta decay per calendar day).
    Tracks how dealer delta hedges erode intraday — critical for 0DTE and
    end-of-week setups where charm can flip the effective GEX sign.

    Formula (annualised):
      call charm = q·exp(-q·T)·N(d1) - exp(-q·T)·n(d1)·[2(r-q)T - d2·σ√T] / (2T·σ√T)
      put  charm = charm_call - q·exp(-q·T)
    Returns per-calendar-day charm (divide annual by 365).
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1v   = _d1(S, K, T, r, sigma, q)
    d2v   = _d2(d1v, sigma, T)
    sqrt_T = math.sqrt(T)
    nd1   = _n(d1v)
    inner = (2 * (r - q) * T - d2v * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    if option_type == "call":
        charm_annual = q * math.exp(-q * T) * _N(d1v) - math.exp(-q * T) * nd1 * inner
    else:
        charm_annual = -q * math.exp(-q * T) * _N(-d1v) - math.exp(-q * T) * nd1 * inner
    return charm_annual / 365.0


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    """Standard BSM delta (call: 0→1, put: -1→0)."""
    if T <= 0 or sigma <= 0:
        return (1.0 if S >= K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
    d1v = _d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return math.exp(-q * T) * _N(d1v)
    return math.exp(-q * T) * (_N(d1v) - 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Vectorised batch variant (NumPy) — for use in scanner hot-path
# ─────────────────────────────────────────────────────────────────────────────

def vec_gamma(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    ivs: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """
    Vectorised gamma for an array of strikes at a common spot/T/r.
    ivs must be same shape as strikes.
    """
    safe_T   = max(T, 1e-6)
    safe_S   = max(S, 1e-6)
    safe_iv  = np.maximum(ivs, 1e-6)
    log_sk   = np.log(safe_S / np.maximum(strikes, 1e-6))
    d1       = (log_sk + (r - q + 0.5 * safe_iv ** 2) * safe_T) / (safe_iv * math.sqrt(safe_T))
    nd1      = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    gamma    = np.exp(-q * safe_T) * nd1 / (safe_S * safe_iv * math.sqrt(safe_T))
    return gamma


def vec_vanna(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    ivs: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """Vectorised vanna = -exp(-q·T)·n(d1)·d2/σ for an array of strikes."""
    safe_T  = max(T, 1e-6)
    safe_S  = max(S, 1e-6)
    safe_iv = np.maximum(ivs, 1e-6)
    sqrtT   = math.sqrt(safe_T)
    log_sk  = np.log(safe_S / np.maximum(strikes, 1e-6))
    d1      = (log_sk + (r - q + 0.5 * safe_iv ** 2) * safe_T) / (safe_iv * sqrtT)
    d2      = d1 - safe_iv * sqrtT
    nd1     = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    return -math.exp(-q * safe_T) * nd1 * d2 / safe_iv


def vec_delta(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    ivs: np.ndarray,
    option_types: np.ndarray,   # array of str "call"/"put"
    q: float = 0.0,
) -> np.ndarray:
    """Vectorised delta for mixed call/put array."""
    from scipy.special import ndtr as _ndtr_sp
    safe_T  = max(T, 1e-6)
    safe_S  = max(S, 1e-6)
    safe_iv = np.maximum(ivs, 1e-6)
    sqrtT   = math.sqrt(safe_T)
    log_sk  = np.log(safe_S / np.maximum(strikes, 1e-6))
    d1      = (log_sk + (r - q + 0.5 * safe_iv ** 2) * safe_T) / (safe_iv * sqrtT)
    # Use erf-based CDF (no scipy needed)
    Nd1     = 0.5 * (1 + np.vectorize(math.erf)(d1 / math.sqrt(2)))
    delta   = math.exp(-q * safe_T) * Nd1
    # Put delta = call delta - 1
    is_put  = np.array([o == "put" for o in option_types], dtype=float)
    return delta - is_put


def implied_vol_approx(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Newton-Raphson IV solver."""
    if T <= 0 or market_price <= 0:
        return 0.0
    sigma = 0.3   # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type, q)
        vega  = bs_vega(S, K, T, r, sigma, q) * 100   # undo the 0.01 scaling
        diff  = price - market_price
        if abs(diff) < tol:
            break
        if abs(vega) < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(1e-4, min(sigma, 5.0))   # clamp
    return sigma
