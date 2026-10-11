"""Tests for strategy_validation.gates.

The Monte Carlo class is the point of the module: the best of 500 pure-noise
strategies must NOT pass the gate when its true trial count is declared. If it
does, the gate is wrong -- fix the gate, not the test.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.stats import norm

from strategy_validation.gates import (
    PerformanceStats,
    StrategyCandidate,
    ValidationLimits,
    deflated_sharpe,
    evaluate,
    expected_max_sharpe,
)

PPY = 252
GAMMA = 0.5772156649


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def stats_from_returns(r: np.ndarray, ppy: int = PPY) -> PerformanceStats:
    """Build PerformanceStats from a per-period return series.

    Every period counts as one 'trade' (fine for a synthetic noise series).
    """
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((peak - equity) / peak).max() * 100.0)
    return PerformanceStats(
        sharpe=float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)),
        max_drawdown_pct=max_dd,
        hit_rate=float((r > 0).mean()),
        n_trades=len(r),
        n_obs=len(r),
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
    )


def strong_stats(**overrides) -> PerformanceStats:
    base = dict(
        sharpe=2.0,
        max_drawdown_pct=8.0,
        hit_rate=0.60,
        n_trades=200,
        n_obs=756,  # three years of daily returns
        skew=0.0,
        kurtosis=3.0,
    )
    base.update(overrides)
    return PerformanceStats(**base)


def candidate(oos=None, n_trials=5, **kwargs) -> StrategyCandidate:
    return StrategyCandidate(
        in_sample=strong_stats(sharpe=2.5),
        out_of_sample=strong_stats() if oos is None else oos,
        n_trials=n_trials,
        **kwargs,
    )


LIMITS = ValidationLimits()


# ---------------------------------------------------------------------------
# expected_max_sharpe
# ---------------------------------------------------------------------------


class TestExpectedMaxSharpe:
    def test_matches_published_formula(self):
        for n in (2, 10, 100, 500):
            want = 0.3 * (
                (1 - GAMMA) * norm.ppf(1 - 1 / n)
                + GAMMA * norm.ppf(1 - 1 / (n * math.e))
            )
            assert expected_max_sharpe(n, 0.09) == pytest.approx(want, rel=1e-9)

    def test_close_to_true_expected_max_of_normals(self):
        # E[max of 500 iid N(0,1)] is ~3.04; the closed form should be within 2%.
        rng = np.random.default_rng(0)
        true_emax = rng.standard_normal((20000, 500)).max(axis=1).mean()
        assert expected_max_sharpe(500, 1.0) == pytest.approx(true_emax, rel=0.02)

    def test_single_trial_has_no_selection_bias(self):
        assert expected_max_sharpe(1, 0.5) == 0.0
        assert expected_max_sharpe(0, 0.5) == 0.0

    def test_increases_with_trials(self):
        vals = [expected_max_sharpe(n, 0.01) for n in (2, 5, 10, 100, 1000, 10**6)]
        assert vals == sorted(vals)
        assert len(set(vals)) == len(vals)

    def test_scales_with_sqrt_variance(self):
        assert expected_max_sharpe(50, 4.0) == pytest.approx(
            2.0 * expected_max_sharpe(50, 1.0)
        )

    def test_rejects_negative_variance(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(10, -0.1)


# ---------------------------------------------------------------------------
# deflated_sharpe
# ---------------------------------------------------------------------------


class TestDeflatedSharpe:
    # per-period SR 0.126 ~ annualized 2.0; variance 1/(n-1) is the null variance
    SR, N_OBS, VAR = 0.126, 756, 1 / 755

    def dsr(self, **kw):
        args = dict(
            observed_sr=self.SR,
            n_trials=10,
            n_obs=self.N_OBS,
            skew=0.0,
            kurtosis=3.0,
            sr_variance=self.VAR,
        )
        args.update(kw)
        return deflated_sharpe(**args)

    def test_is_a_probability(self):
        for n in (1, 2, 50, 10**5):
            for sr in (-0.2, 0.0, 0.05, 0.3):
                assert 0.0 <= self.dsr(n_trials=n, observed_sr=sr) <= 1.0

    def test_monotone_decreasing_in_trials(self):
        vals = [self.dsr(n_trials=n) for n in (1, 2, 5, 20, 100, 1000, 10**5)]
        assert all(a > b for a, b in zip(vals, vals[1:])), vals

    def test_monotone_increasing_in_observed_sharpe(self):
        # The viral variant fails exactly here (passes 3.0, rejects 1.8).
        annualized = [0.0, 0.5, 1.0, 1.8, 2.0, 3.0, 4.0]
        vals = [self.dsr(observed_sr=s / math.sqrt(PPY), n_trials=100) for s in annualized]
        assert all(a < b for a, b in zip(vals, vals[1:])), vals

    def test_higher_sharpe_beats_lower_at_the_viral_counterexample(self):
        low = self.dsr(observed_sr=1.8 / math.sqrt(PPY), n_trials=100)
        high = self.dsr(observed_sr=3.0 / math.sqrt(PPY), n_trials=100)
        assert high > low

    def test_longer_sample_helps_when_sharpe_beats_benchmark(self):
        short = self.dsr(n_obs=252, sr_variance=1 / 251)
        long_ = self.dsr(n_obs=2520, sr_variance=1 / 2519)
        assert long_ > short

    def test_zero_sharpe_single_trial_is_a_coin_flip(self):
        assert self.dsr(observed_sr=0.0, n_trials=1) == pytest.approx(0.5)

    def test_matches_direct_computation(self):
        sr0 = expected_max_sharpe(30, self.VAR)
        se = math.sqrt((1 - 0.1 * self.SR + (4.0 - 1) / 4 * self.SR**2) / (self.N_OBS - 1))
        want = norm.cdf((self.SR - sr0) / se)
        got = self.dsr(n_trials=30, skew=0.1, kurtosis=4.0)
        assert got == pytest.approx(want, rel=1e-9)

    def test_negative_skew_and_fat_tails_lower_the_score(self):
        normal = self.dsr(n_trials=30)
        ugly = self.dsr(n_trials=30, skew=-1.0, kurtosis=8.0)
        assert ugly < normal

    def test_fails_closed_on_bad_inputs(self):
        assert self.dsr(n_obs=1) == 0.0
        assert self.dsr(observed_sr=float("nan")) == 0.0
        assert self.dsr(sr_variance=float("nan")) == 0.0
        # degenerate standard-error term (violates kurtosis >= skew^2 + 1)
        assert self.dsr(skew=50.0, kurtosis=1.0, observed_sr=0.5) == 0.0


# ---------------------------------------------------------------------------
# evaluate: precedence + each branch
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_limits_are_frozen_with_spec_defaults(self):
        lim = ValidationLimits()
        assert (lim.min_oos_sharpe, lim.max_drawdown_pct, lim.min_hit_rate) == (1.0, 15.0, 0.55)
        assert (lim.min_trades, lim.min_dsr, lim.require_oos) == (30, 0.95, True)
        with pytest.raises(Exception):
            lim.min_dsr = 0.0  # type: ignore[misc]

    def test_strong_strategy_is_approved(self):
        v = evaluate(candidate(n_trials=5), LIMITS)
        assert v.approved is True
        assert v.reason == "approved"
        assert v.metrics["dsr"] >= 0.95

    def test_missing_oos(self):
        c = StrategyCandidate(in_sample=strong_stats(), out_of_sample=None, n_trials=5)
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "missing_oos")

    def test_missing_oos_when_a_field_is_none_or_nan(self):
        for field in ("sharpe", "n_trades", "n_obs", "max_drawdown_pct", "hit_rate"):
            for bad in (None, float("nan")):
                v = evaluate(candidate(oos=strong_stats(**{field: bad})), LIMITS)
                assert (v.approved, v.reason) == (False, "missing_oos"), (field, bad)

    def test_in_sample_alone_cannot_approve(self):
        c = StrategyCandidate(in_sample=strong_stats(), out_of_sample=None, n_trials=1)
        assert evaluate(c, LIMITS).reason == "missing_oos"

    def test_require_oos_false_gates_on_in_sample_with_warning(self):
        c = StrategyCandidate(in_sample=strong_stats(), out_of_sample=None, n_trials=5)
        v = evaluate(c, ValidationLimits(require_oos=False))
        assert v.approved is True
        assert any("IN-SAMPLE" in w for w in v.metrics["warnings"])

    def test_insufficient_trades(self):
        v = evaluate(candidate(oos=strong_stats(n_trades=29)), LIMITS)
        assert (v.approved, v.reason) == (False, "insufficient_trades")
        assert evaluate(candidate(oos=strong_stats(n_trades=30)), LIMITS).approved

    def test_drawdown(self):
        v = evaluate(candidate(oos=strong_stats(max_drawdown_pct=15.01)), LIMITS)
        assert (v.approved, v.reason) == (False, "drawdown")
        assert evaluate(candidate(oos=strong_stats(max_drawdown_pct=15.0)), LIMITS).approved

    def test_drawdown_sign_convention_is_ignored(self):
        assert evaluate(candidate(oos=strong_stats(max_drawdown_pct=-8.0)), LIMITS).approved
        v = evaluate(candidate(oos=strong_stats(max_drawdown_pct=-20.0)), LIMITS)
        assert v.reason == "drawdown"

    def test_hit_rate(self):
        v = evaluate(candidate(oos=strong_stats(hit_rate=0.549)), LIMITS)
        assert (v.approved, v.reason) == (False, "hit_rate")

    def test_hit_rate_given_as_percent_is_rejected_not_passed(self):
        v = evaluate(candidate(oos=strong_stats(hit_rate=60.0)), LIMITS)
        assert (v.approved, v.reason) == (False, "hit_rate")

    def test_oos_sharpe(self):
        v = evaluate(candidate(oos=strong_stats(sharpe=0.99)), LIMITS)
        assert (v.approved, v.reason) == (False, "oos_sharpe")

    @pytest.mark.parametrize("bad", [None, 0, -3])
    def test_undeclared_trial_count(self, bad):
        v = evaluate(candidate(n_trials=bad), LIMITS)
        assert (v.approved, v.reason) == (False, "undeclared_trial_count")

    def test_deflated_sharpe(self):
        # Clears every cheap gate (Sharpe 1.2) but not after 10_000 trials.
        c = candidate(oos=strong_stats(sharpe=1.2), n_trials=10_000)
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "deflated_sharpe")
        assert v.metrics["dsr"] < 0.95

    def test_ledger_larger_than_declared_n_trials_is_rejected(self):
        c = candidate(n_trials=3, trial_sharpes=[0.1] * 10)
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "undeclared_trial_count")

    def test_n_trials_one_without_ledger_is_rejected(self):
        # n_trials=1 switches the whole correction off and cannot be verified from
        # the numbers, so it fails closed unless a ledger backs it.
        for ledger in (None, []):
            v = evaluate(candidate(n_trials=1, trial_sharpes=ledger), LIMITS)
            assert (v.approved, v.reason) == (False, "undeclared_trial_count"), ledger
            assert "ledger" in v.metrics["detail"]

    def test_genuine_single_hypothesis_passes_with_a_one_element_ledger(self):
        # The honest path: an explicit, cheap act. SR0 is 0 for one trial.
        v = evaluate(candidate(n_trials=1, trial_sharpes=[2.0]), LIMITS)
        assert v.approved is True, v
        assert v.metrics["benchmark_sharpe_per_period"] == 0.0

    def test_one_element_ledger_does_not_license_a_larger_search(self):
        # Ledger longer than the declared n_trials is still an understatement.
        v = evaluate(candidate(n_trials=1, trial_sharpes=[2.0, 1.1]), LIMITS)
        assert (v.approved, v.reason) == (False, "undeclared_trial_count")

    def test_first_failure_wins(self):
        # Everything is wrong at once; order must be missing_oos first...
        everything_bad = strong_stats(
            sharpe=0.1, max_drawdown_pct=50.0, hit_rate=0.3, n_trades=1
        )
        c = StrategyCandidate(in_sample=everything_bad, out_of_sample=None, n_trials=None)
        assert evaluate(c, LIMITS).reason == "missing_oos"
        # ...then trades -> drawdown -> hit_rate -> oos_sharpe -> trial count -> DSR.
        order = [
            ("insufficient_trades", dict(n_trades=1, max_drawdown_pct=50.0, hit_rate=0.3, sharpe=0.1)),
            ("drawdown", dict(max_drawdown_pct=50.0, hit_rate=0.3, sharpe=0.1)),
            ("hit_rate", dict(hit_rate=0.3, sharpe=0.1)),
            ("oos_sharpe", dict(sharpe=0.1)),
        ]
        for reason, kw in order:
            v = evaluate(candidate(oos=strong_stats(**kw), n_trials=None), LIMITS)
            assert v.reason == reason, (reason, v)
        # trial-count check precedes the DSR check
        assert evaluate(candidate(n_trials=None), LIMITS).reason == "undeclared_trial_count"

    def test_pure_no_mutation_of_inputs(self):
        c = candidate()
        before = (c.out_of_sample.sharpe, c.n_trials, c.in_sample.sharpe)
        evaluate(c, LIMITS)
        assert before == (c.out_of_sample.sharpe, c.n_trials, c.in_sample.sharpe)


# ---------------------------------------------------------------------------
# The point of the module: best-of-N noise must not pass.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def noise_search():
    """500 pure-noise strategies, 500 daily returns each, seed 42."""
    rng = np.random.default_rng(42)
    series = rng.normal(0, 0.01, (500, 500))
    all_stats = [stats_from_returns(r) for r in series]
    sharpes = np.array([s.sharpe for s in all_stats])
    best = int(np.argmax(sharpes))
    return series[best], all_stats[best], sharpes


def margin_noise_stats() -> PerformanceStats:
    """A deliberately constructed 'best-of-500 noise'-style candidate.

    Annualized Sharpe 2.4 over 500 daily observations is what the best of 500
    noise series reaches (the seed-42 winner has 2.379). Every other gate is
    cleared with a wide margin (hit rate 0.58 vs 0.55, drawdown 9% vs 15%, 500
    trades vs 30, Sharpe 2.4 vs 1.0), so the ONLY rule that can fire is the
    deflated-Sharpe rule. Unlike the real seed-42 winner it does not sit on a
    threshold boundary.
    """
    return PerformanceStats(
        sharpe=2.4,
        max_drawdown_pct=9.0,
        hit_rate=0.58,
        n_trades=500,
        n_obs=500,
        skew=0.0,
        kurtosis=3.0,
    )


class TestMonteCarloNoise:
    def test_best_of_500_noise_looks_great_in_isolation(self, noise_search):
        # Sanity: the selection effect is real, otherwise the test proves nothing.
        # (Hit rate is deliberately NOT asserted here: the seed-42 winner lands
        # exactly on 0.550, the gate boundary -- see the next test.)
        _, best, _ = noise_search
        assert best.sharpe > 2.0  # pure noise, annualized
        assert best.max_drawdown_pct < 15.0

    def test_best_of_500_noise_is_rejected(self, noise_search):
        # The search reused the same window to rank all 500 strategies, so it is
        # not a clean hold-out: the best strategy's window is both the selection
        # and the "OOS" evidence. That is the realistic 24/7-search failure mode.
        #
        # What matters is `approved is False`. The reason is allowed to be either
        # "deflated_sharpe" (the multiple-testing rule, the intended catch) or
        # "hit_rate": the seed-42 winner's hit rate is exactly 0.550, sitting ON
        # the `>= 0.55` boundary, so if the RNG stream ever changed and nudged it
        # to 0.548 the hit-rate rule would reject it first. That would be the gate
        # working, not breaking. The exact-reason pin lives in
        # test_noise_like_candidate_is_rejected_by_deflated_sharpe_exactly, which
        # uses inputs with margin on every other gate.
        _, best, _ = noise_search
        c = StrategyCandidate(in_sample=best, out_of_sample=best, n_trials=500)
        v = evaluate(c, LIMITS)
        print(
            f"\nbest-of-500 noise: annualized Sharpe {best.sharpe:.3f}, "
            f"hit {best.hit_rate:.3f}, dd {best.max_drawdown_pct:.2f}%, "
            f"DSR {v.metrics.get('dsr', float('nan')):.4f} -> {v.reason}"
        )
        assert v.approved is False
        assert v.reason in {"deflated_sharpe", "hit_rate"}, v
        if v.reason == "deflated_sharpe":
            assert v.metrics["dsr"] < 0.95

    def test_noise_like_candidate_is_rejected_by_deflated_sharpe_exactly(self):
        # Clears every other gate with margin, so the DSR rule must be the one
        # that fires. Noise-level Sharpe 2.4, true trial count 500.
        c = StrategyCandidate(
            in_sample=margin_noise_stats(),
            out_of_sample=margin_noise_stats(),
            n_trials=500,
        )
        v = evaluate(c, LIMITS)
        assert v.approved is False
        assert v.reason == "deflated_sharpe"
        assert v.metrics["dsr"] < 0.80  # nowhere near the 0.95 bar
        # ...and it is exactly the trial count that does it: the same numbers
        # backed by a one-element ledger (a "single hypothesis") would pass.
        honest_single = StrategyCandidate(
            in_sample=margin_noise_stats(),
            out_of_sample=margin_noise_stats(),
            n_trials=1,
            trial_sharpes=[2.4],
        )
        assert evaluate(honest_single, LIMITS).approved is True

    def test_ledger_supplied_variance_still_rejects(self, noise_search):
        # Variance comes from the real 500-strategy search ledger.
        _, _, sharpes = noise_search
        c = StrategyCandidate(
            in_sample=margin_noise_stats(),
            out_of_sample=margin_noise_stats(),
            n_trials=500,
            trial_sharpes=list(sharpes),
        )
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "deflated_sharpe")
        assert v.metrics["sr_variance_source"] == "trial_sharpes ledger"

    def test_dishonest_n_trials_with_ledger_is_rejected(self, noise_search):
        # Same candidate, lying n_trials=1, but the search ledger gives it away.
        _, _, sharpes = noise_search
        c = StrategyCandidate(
            in_sample=margin_noise_stats(),
            out_of_sample=margin_noise_stats(),
            n_trials=1,
            trial_sharpes=list(sharpes),
        )
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "undeclared_trial_count")

    def test_dishonest_n_trials_without_ledger_is_rejected(self, noise_search):
        # The case the old gate approved with only a warning (DSR 0.9996 at N=1).
        # It now fails closed: n_trials=1 with no ledger cannot be verified and
        # is the claim that disables the correction.
        c = StrategyCandidate(
            in_sample=margin_noise_stats(),
            out_of_sample=margin_noise_stats(),
            n_trials=1,
        )
        v = evaluate(c, LIMITS)
        assert (v.approved, v.reason) == (False, "undeclared_trial_count")
        # The real seed-42 winner too. Reason may be hit_rate if the boundary
        # flips, but it must never be approved.
        _, best, _ = noise_search
        real = StrategyCandidate(in_sample=best, out_of_sample=best, n_trials=1)
        rv = evaluate(real, LIMITS)
        assert rv.approved is False
        assert rv.reason in {"undeclared_trial_count", "hit_rate"}, rv

    def test_true_holdout_catches_noise_even_before_the_dsr(self, noise_search):
        # Honest OOS separation: rank 500 noise strategies on the first half,
        # evaluate the winner on a fresh second half. The edge evaporates.
        rng = np.random.default_rng(42)
        series = rng.normal(0, 0.01, (500, 500))
        is_stats = [stats_from_returns(r[:250]) for r in series]
        best = int(np.argmax([s.sharpe for s in is_stats]))
        oos = stats_from_returns(np.random.default_rng(7).normal(0, 0.01, 250))
        c = StrategyCandidate(in_sample=is_stats[best], out_of_sample=oos, n_trials=500)
        v = evaluate(c, LIMITS)
        assert is_stats[best].sharpe > 2.5  # looked fantastic in-sample
        assert v.approved is False

    def test_many_noise_winners_never_pass_at_true_n(self):
        # 20 independent searches of 100 noise strategies each: none may pass.
        rng = np.random.default_rng(123)
        passed = 0
        for _ in range(20):
            series = rng.normal(0, 0.01, (100, 500))
            all_stats = [stats_from_returns(r) for r in series]
            best = max(all_stats, key=lambda s: s.sharpe)
            c = StrategyCandidate(in_sample=best, out_of_sample=best, n_trials=100)
            passed += evaluate(c, LIMITS).approved
        assert passed == 0

    def test_genuinely_strong_strategy_is_approved(self):
        # OOS Sharpe 2.0, drawdown 8%, hit rate 0.60, 200 trades, n_trials=5.
        v = evaluate(candidate(n_trials=5), LIMITS)
        assert v.approved is True, v
        assert v.metrics["dsr"] >= LIMITS.min_dsr

    def test_strong_strategy_is_still_deflated_by_a_huge_search(self):
        # The same strategy, if it was the best of a million trials, is not credible.
        v = evaluate(candidate(n_trials=1_000_000), LIMITS)
        assert (v.approved, v.reason) == (False, "deflated_sharpe")
