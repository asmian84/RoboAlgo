"""
gates — pure strategy-validation gate with a multiple-testing correction.

ZERO I/O: no DB, no network, no file reads. Plain data in, verdict out. Mirrors
the shape of auto_trader/governor.py (frozen limits dataclass, rules in strict
precedence order, first failure wins, fully unit-testable).

WHY THIS EXISTS, AND HOW IT DIVERGES FROM ITS SOURCE
----------------------------------------------------
The idea came from an X thread ("How to Use GPT-6 Astra to Find Profitable
Trading Strategies 24/7", @RohOnChain). It describes an agent that searches for
strategies around the clock and lets one trade only if it passes: Sharpe > 1.5,
max drawdown < 15%, hit rate > 55%, t-stat > 2.0.

What is ADOPTED from the thread: the *structure* -- an explicit, enforced barrier
that a strategy must clear before it is allowed to trade.

What is REJECTED: the thresholds as stated. They are applied to the BEST of an
unbounded, 24/7 search with NO multiple-testing correction. Picking the best of N
strategies on pure noise inflates Sharpe by construction: best-of-10 noise
already has a median Sharpe around 1.0 (Monte Carlo, earlier in this project),
and a search that never stops runs effectively unbounded trials. Gates like
"Sharpe > 1.5 and t > 2" on the winner of that search are the textbook
overfitting trap, not a safeguard.

What is ADDED:
  1. In-sample / out-of-sample separation. Gated metrics come from OOS data
     (``require_oos=True`` by default). In-sample numbers are recorded but a
     strategy cannot be approved on them.
  2. A trial-count correction: the Deflated Sharpe Ratio (Bailey & Lopez de
     Prado, "The Deflated Sharpe Ratio", 2014). The observed Sharpe is compared
     with the Sharpe the best of ``n_trials`` skill-less strategies would be
     expected to reach, and the result is a probability that the true Sharpe is
     above zero after that selection bias.
  3. A hard requirement that ``n_trials`` be declared. A search that will not
     report how many candidates it tried cannot be validated, so it is rejected.

KNOWN-BROKEN VARIANT: a widely circulated "viral" version of the DSR formula is
wrong. It was tested earlier in this project and it both passes a Sharpe of 3.0
and rejects a Sharpe of 1.8 (i.e. it is not monotone in Sharpe). Do not replace
the implementation below with it. This module uses the Bailey & Lopez de Prado
original and the tests assert monotonicity (more trials -> lower DSR, higher
Sharpe -> higher DSR).

HONEST LIMITS OF THE GATE
-------------------------
* ``n_trials`` is self-reported by the caller. A pure function cannot detect a
  lie about it from the numbers alone: a best-of-500 noise series scores
  DSR ~0.58 with n_trials=500 but DSR ~0.9996 with n_trials=1. Two mitigations:
  (a) a candidate may carry ``trial_sharpes`` (the Sharpe of every strategy that
  was searched); if ``n_trials`` is smaller than the length of that ledger the
  candidate is rejected as ``undeclared_trial_count``; (b) the gate FAILS CLOSED
  on ``n_trials <= 1`` unless a ledger is present (see next section).
  Callers should take ``n_trials`` from an append-only search ledger, never
  from the agent's own claim.

WHY ``n_trials == 1`` WITHOUT A LEDGER IS REJECTED
--------------------------------------------------
This gate exists to validate candidates produced by an automated search that
runs continuously. In that setting "I tried exactly one hypothesis" is the
single most consequential claim a caller can make: it is the claim that
switches the whole selection-bias correction off (the benchmark Sharpe becomes
0), and it is unverifiable from the numbers. Approving it by default would make
the one input that matters most the one input nobody checks. So it is rejected
as ``undeclared_trial_count`` unless the candidate carries a ``trial_sharpes``
ledger. A genuine pre-registered single hypothesis still passes: it supplies a
1-element ledger, which makes the honest path an explicit, cheap act and the
dishonest path impossible by default. Fail closed on the input that disables the
safety mechanism.
* Passing this gate is necessary, not sufficient. It says nothing about
  transaction costs, capacity, regime change, or look-ahead bugs in the backtest.
* The OOS window is deflated by ``n_trials`` on the conservative assumption that
  every one of the searched candidates was looked at on it (OOS windows that are
  re-used across a search stop being out-of-sample).

UNITS
-----
All Sharpe-like inputs on ``PerformanceStats`` / ``StrategyCandidate`` are
ANNUALIZED, matching ``ValidationLimits.min_oos_sharpe``. The Deflated Sharpe
math (``expected_max_sharpe`` / ``deflated_sharpe``) is defined on PER-PERIOD
(non-annualized) Sharpe ratios, so ``evaluate`` divides by
``sqrt(periods_per_year)`` before calling it. Kurtosis is the RAW kurtosis
(normal distribution = 3.0), not excess kurtosis.

Rules are evaluated in strict precedence order (first failure wins):
    1. missing_oos
    2. insufficient_trades
    3. drawdown
    4. hit_rate
    5. oos_sharpe
    6. undeclared_trial_count  (n_trials None / < 1, understated vs. ledger,
                                or n_trials == 1 with no trial_sharpes ledger)
    7. deflated_sharpe
    else -> approved
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014). Per-period units.
# ---------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe of ``n_trials`` skill-less strategies.

    Bailey & Lopez de Prado:
        sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N*e)))
    with g the Euler-Mascheroni constant and Phi^-1 the standard normal
    quantile. This is the Sharpe you should expect to "find" by selection alone,
    i.e. the benchmark an observed Sharpe has to beat.

    ``sr_variance`` is the variance of the Sharpe estimates across the trials,
    in the same (per-period) units as the Sharpe being deflated.

    With ``n_trials <= 1`` there is no selection, so the benchmark is 0.0.
    """
    if sr_variance < 0 or math.isnan(sr_variance):
        raise ValueError(f"sr_variance must be >= 0, got {sr_variance}")
    if n_trials <= 1:
        return 0.0
    g = EULER_MASCHERONI
    n = float(n_trials)
    return math.sqrt(sr_variance) * (
        (1.0 - g) * norm.ppf(1.0 - 1.0 / n)
        + g * norm.ppf(1.0 - 1.0 / (n * math.e))
    )


def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_variance: float,
) -> float:
    """Deflated Sharpe Ratio: P(true Sharpe > 0 | selected best of n_trials).

    All Sharpe inputs are PER-PERIOD (non-annualized); ``kurtosis`` is RAW
    (normal = 3.0). Computes

        DSR = Phi( (SR - SR0) / se(SR) )
        SR0 = expected_max_sharpe(n_trials, sr_variance)
        se(SR) = sqrt((1 - skew*SR + ((kurtosis - 1)/4) * SR^2) / (n_obs - 1))

    which is the Bailey & Lopez de Prado original with the non-normal Sharpe
    standard error. (Do NOT substitute the viral variant -- it is not monotone
    in Sharpe; see the module docstring.)

    Fails closed: returns 0.0 if the sample is too short (``n_obs < 2``), any
    input is NaN, or the standard-error term is not positive.
    """
    values = (observed_sr, skew, kurtosis, sr_variance)
    if any(v is None or math.isnan(v) for v in values):
        return 0.0
    if n_obs is None or n_obs < 2:
        return 0.0

    sr0 = expected_max_sharpe(n_trials, sr_variance)
    var_term = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if var_term <= 0.0:
        return 0.0
    se = math.sqrt(var_term / (n_obs - 1))
    return float(norm.cdf((observed_sr - sr0) / se))


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationLimits:
    min_oos_sharpe: float = 1.0
    max_drawdown_pct: float = 15.0
    min_hit_rate: float = 0.55
    min_trades: int = 30
    min_dsr: float = 0.95
    require_oos: bool = True


@dataclass
class PerformanceStats:
    """Performance of one strategy over one data window.

    sharpe is ANNUALIZED. max_drawdown_pct is a magnitude in percent (15.0 means
    15%; the sign is ignored). hit_rate is a fraction in [0, 1]. kurtosis is RAW
    (normal = 3.0). n_obs is the number of return observations in the window.
    """

    sharpe: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    hit_rate: Optional[float] = None
    n_trades: Optional[int] = None
    n_obs: Optional[int] = None
    skew: float = 0.0
    kurtosis: float = 3.0


@dataclass
class StrategyCandidate:
    """A strategy plus both of its evaluation windows and its search history.

    n_trials: how many candidates were searched to produce this one (including
        this one). None / < 1 is rejected as ``undeclared_trial_count``; so is
        n_trials == 1 unless ``trial_sharpes`` is supplied (a pre-registered
        single hypothesis passes with a 1-element ledger).
    trial_sharpes: optional ledger -- the ANNUALIZED Sharpe of every searched
        candidate. If given, it (a) cross-checks ``n_trials`` and (b) supplies
        the cross-trial Sharpe variance for the DSR benchmark.
    trial_sharpe_variance: optional ANNUALIZED cross-trial Sharpe variance, used
        when no ledger is given. If neither is given the null-hypothesis
        variance 1/(n_obs - 1) (per-period) is used.
    """

    in_sample: PerformanceStats
    out_of_sample: Optional[PerformanceStats] = None
    n_trials: Optional[int] = None
    name: str = ""
    periods_per_year: int = 252
    trial_sharpes: Optional[Sequence[float]] = None
    trial_sharpe_variance: Optional[float] = None


@dataclass
class Verdict:
    approved: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _absent(value: Any) -> bool:
    """True for None or NaN. NaN must never slip through a ``<`` comparison."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _reject(reason: str, detail: str, metrics: dict[str, Any]) -> Verdict:
    metrics["detail"] = detail
    return Verdict(False, reason, metrics)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def evaluate(candidate: StrategyCandidate, limits: ValidationLimits) -> Verdict:
    """Run the gate. First failing rule wins; see the module docstring for order.

    The gated window is out-of-sample. With ``limits.require_oos=False`` (research
    mode only -- never use it to authorize live trading) the in-sample window is
    used when no OOS window is supplied.
    """
    metrics: dict[str, Any] = {
        "name": candidate.name,
        "n_trials": candidate.n_trials,
        "in_sample_sharpe": candidate.in_sample.sharpe,
        "warnings": [],
    }
    oos = candidate.out_of_sample
    window_name = "out_of_sample"

    # 1. missing_oos
    if oos is None and not limits.require_oos:
        oos = candidate.in_sample
        window_name = "in_sample"
        metrics["warnings"].append(
            "require_oos=False: gated on IN-SAMPLE data; research use only"
        )
    required = (
        ("sharpe", "n_trades", "n_obs", "max_drawdown_pct", "hit_rate")
        if oos is not None
        else ()
    )
    missing = [f for f in required if _absent(getattr(oos, f))]
    if oos is None or missing:
        return _reject(
            "missing_oos",
            "no out-of-sample window supplied"
            if oos is None
            else f"{window_name} metrics missing/NaN: {', '.join(missing)}",
            metrics,
        )
    if oos.n_obs < 2:
        return _reject("missing_oos", f"{window_name} n_obs {oos.n_obs} < 2", metrics)

    metrics.update(
        window=window_name,
        sharpe=oos.sharpe,
        max_drawdown_pct=abs(oos.max_drawdown_pct),
        hit_rate=oos.hit_rate,
        n_trades=oos.n_trades,
        n_obs=oos.n_obs,
    )
    if candidate.in_sample.sharpe is not None and not _absent(candidate.in_sample.sharpe):
        if candidate.in_sample.sharpe > 0:
            metrics["oos_to_is_sharpe_ratio"] = oos.sharpe / candidate.in_sample.sharpe

    # 2. insufficient_trades
    if not (oos.n_trades >= limits.min_trades):
        return _reject(
            "insufficient_trades",
            f"n_trades {oos.n_trades} < min_trades {limits.min_trades}",
            metrics,
        )

    # 3. drawdown (magnitude)
    drawdown = abs(oos.max_drawdown_pct)
    if not (drawdown <= limits.max_drawdown_pct):
        return _reject(
            "drawdown",
            f"max_drawdown {drawdown:.2f}% > max_drawdown_pct "
            f"{limits.max_drawdown_pct:.2f}%",
            metrics,
        )

    # 4. hit_rate
    if not (0.0 <= oos.hit_rate <= 1.0) or not (oos.hit_rate >= limits.min_hit_rate):
        return _reject(
            "hit_rate",
            f"hit_rate {oos.hit_rate:.4f} < min_hit_rate {limits.min_hit_rate:.4f} "
            "(or outside [0, 1])",
            metrics,
        )

    # 5. oos_sharpe
    if not (oos.sharpe >= limits.min_oos_sharpe):
        return _reject(
            "oos_sharpe",
            f"{window_name} sharpe {oos.sharpe:.3f} < min_oos_sharpe "
            f"{limits.min_oos_sharpe:.3f}",
            metrics,
        )

    # 6. undeclared_trial_count
    n_trials = candidate.n_trials
    if _absent(n_trials) or n_trials < 1:
        return _reject(
            "undeclared_trial_count",
            f"n_trials={n_trials!r}: a search that does not report how many "
            "candidates it tried cannot be validated",
            metrics,
        )
    ledger = candidate.trial_sharpes
    has_ledger = ledger is not None and len(ledger) >= 1
    if ledger is not None and len(ledger) > n_trials:
        return _reject(
            "undeclared_trial_count",
            f"declared n_trials={n_trials} but the search ledger holds "
            f"{len(ledger)} trial Sharpes; n_trials is understated",
            metrics,
        )
    if n_trials <= 1 and not has_ledger:
        # n_trials == 1 turns the selection-bias correction off entirely and is
        # unverifiable from the numbers: fail closed unless a ledger backs it.
        return _reject(
            "undeclared_trial_count",
            f"n_trials={n_trials} disables the multiple-testing correction and "
            "no trial_sharpes ledger backs it; supply a ledger (a 1-element "
            "ledger is enough for a genuine single pre-registered hypothesis)",
            metrics,
        )

    # 7. deflated_sharpe -- all in per-period units
    scale = math.sqrt(candidate.periods_per_year)
    sr_pp = oos.sharpe / scale
    if ledger is not None and len(ledger) >= 2:
        sr_variance = _sample_variance([s / scale for s in ledger])
        variance_source = "trial_sharpes ledger"
    elif not _absent(candidate.trial_sharpe_variance):
        sr_variance = candidate.trial_sharpe_variance / candidate.periods_per_year
        variance_source = "trial_sharpe_variance"
    else:
        sr_variance = 1.0 / (oos.n_obs - 1)
        variance_source = "null 1/(n_obs-1)"

    sr0 = expected_max_sharpe(n_trials, sr_variance)
    dsr = deflated_sharpe(
        sr_pp, n_trials, oos.n_obs, oos.skew, oos.kurtosis, sr_variance
    )
    metrics.update(
        dsr=dsr,
        benchmark_sharpe_per_period=sr0,
        benchmark_sharpe_annualized=sr0 * scale,
        sr_variance_per_period=sr_variance,
        sr_variance_source=variance_source,
    )
    if not (dsr >= limits.min_dsr):
        return _reject(
            "deflated_sharpe",
            f"DSR {dsr:.4f} < min_dsr {limits.min_dsr:.4f} with n_trials="
            f"{n_trials} (selection benchmark {sr0 * scale:.3f} annualized vs "
            f"observed {oos.sharpe:.3f})",
            metrics,
        )

    metrics["detail"] = "approved"
    return Verdict(True, "approved", metrics)
