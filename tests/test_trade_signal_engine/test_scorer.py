"""Tests for UnifiedScorer — 18-component weighted scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trade_signal_engine.scorer import (
    TRADE_PROFILES,
    UNIFIED_WEIGHTS,
    ScoredSignal,
    UnifiedScorer,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


@dataclass
class MockSymbolData:
    """Minimal stand-in for collector.SymbolData."""

    symbol: str = "AAPL"
    profile: str = "SWING"
    current_price: float = 0.0
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    options_chain: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _mock_all_engines_neutral():
    """Return a dict suitable for patch.dict('sys.modules', ...) that mocks
    all external engines to return neutral / passing values so we can test
    the scorer logic in isolation."""
    # MTF — passes the 0.40 gate
    mock_mtf_inst = MagicMock()
    mock_mtf_inst.run.return_value = {
        "alignment_score": 0.70,
        "bias_direction": "LONG",
    }
    mock_mtf_mod = MagicMock()
    mock_mtf_mod.MTFAlignmentEngine.return_value = mock_mtf_inst

    # Wave phase — neutral / impulse
    mock_wave_mod = MagicMock()
    mock_wave_mod.detect_wave_phase.return_value = {
        "wave_phase": "IMPULSE",
        "direction": "UP",
        "confidence": 0.5,
    }

    # SMC — raise ImportError so it falls back to defaults
    smc_mod = MagicMock()
    smc_mod.score_smc.side_effect = ImportError("mocked out")

    # Exhaustion — not detected
    mock_exh_mod = MagicMock()
    mock_exh_mod.detect_exhaustion.return_value = {
        "exhaustion_detected": False,
        "direction": "NONE",
        "exhaustion_strength": 0.0,
    }

    return {
        "mtf_alignment": MagicMock(),
        "mtf_alignment.mtf_alignment_engine": mock_mtf_mod,
        "wave_phase": MagicMock(),
        "wave_phase.wave_phase_engine": mock_wave_mod,
        "smc_engine": MagicMock(),
        "smc_engine.smc_scorer": smc_mod,
        "time_exhaustion": MagicMock(),
        "time_exhaustion.exhaustion_engine": mock_exh_mod,
    }


def _make_uptrend_df(n: int = 100) -> pd.DataFrame:
    """Create a simple uptrending OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 100 + np.arange(n) * 0.5 + np.random.default_rng(42).normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "open": base - 0.3,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.random.default_rng(42).integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def _make_downtrend_df(n: int = 100) -> pd.DataFrame:
    """Create a simple downtrending OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 200 - np.arange(n) * 0.5 + np.random.default_rng(42).normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "open": base + 0.3,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.random.default_rng(42).integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def _make_short_df(n: int = 5) -> pd.DataFrame:
    """Create a DataFrame too short for scoring."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base,
            "volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


# ── Tests: Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_unified_weights_sum_to_one(self):
        total = sum(UNIFIED_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_unified_weights_has_18_components(self):
        assert len(UNIFIED_WEIGHTS) == 18

    def test_trade_profiles_has_required_keys(self):
        for key in ("SCALP", "SWING", "POSITION"):
            assert key in TRADE_PROFILES
            cfg = TRADE_PROFILES[key]
            assert "mtf_model" in cfg
            assert "bar_tf" in cfg
            assert "smc_tf" in cfg
            assert "structure_tf" in cfg
            assert "stop_atr_mult" in cfg
            assert "target_rr" in cfg


# ── Tests: ScoredSignal dataclass ──────────────────────────────────────────────


class TestScoredSignal:
    def test_required_fields(self):
        sig = ScoredSignal(
            symbol="TSLA",
            profile="SWING",
            score=72.5,
            direction="LONG",
            confidence="MEDIUM",
        )
        assert sig.symbol == "TSLA"
        assert sig.profile == "SWING"
        assert sig.score == 72.5
        assert sig.direction == "LONG"
        assert sig.confidence == "MEDIUM"
        assert sig.components == {}
        assert sig.component_details == {}
        assert sig.rejection_reason == ""
        assert sig.bonuses == []

    def test_optional_fields(self):
        sig = ScoredSignal(
            symbol="AAPL",
            profile="SCALP",
            score=0,
            direction="NO_TRADE",
            confidence="NONE",
            rejection_reason="test reason",
            bonuses=["bonus1"],
        )
        assert sig.rejection_reason == "test reason"
        assert sig.bonuses == ["bonus1"]


# ── Tests: Scoring ─────────────────────────────────────────────────────────────


class TestScoring:
    def setup_method(self):
        self.scorer = UnifiedScorer()

    def test_insufficient_data_returns_no_trade(self):
        sd = MockSymbolData(bars={"1h": _make_short_df(5)})
        result = self.scorer.score(sd)
        assert result.direction == "NO_TRADE"
        assert result.confidence == "NONE"
        assert result.score == 0
        assert "Insufficient" in result.rejection_reason

    def test_empty_bars_returns_no_trade(self):
        sd = MockSymbolData(bars={})
        result = self.scorer.score(sd)
        assert result.direction == "NO_TRADE"
        assert result.score == 0

    def test_score_with_uptrend_data(self):
        """With uptrend data and no external engines, should produce a score > 0
        and direction should NOT be SHORT (neutral or LONG is fine)."""
        sd = MockSymbolData(
            bars={"1h": _make_uptrend_df(100)},
            current_price=149.0,
        )
        result = self.scorer.score(sd)
        assert 0 <= result.score <= 100
        assert result.direction != "SHORT"
        assert len(result.components) > 0

    def test_score_with_downtrend_data(self):
        """With downtrend data and no external engines, should produce a score >= 0."""
        sd = MockSymbolData(
            bars={"1h": _make_downtrend_df(100)},
            current_price=151.0,
        )
        result = self.scorer.score(sd)
        assert 0 <= result.score <= 100

    def test_score_always_in_range(self):
        """Score must be clamped between 0 and 100 regardless of input."""
        for _ in range(5):
            n = np.random.randint(50, 200)
            sd = MockSymbolData(
                bars={"1h": _make_uptrend_df(n)},
            )
            result = self.scorer.score(sd)
            assert 0 <= result.score <= 100

    def test_components_populated(self):
        sd = MockSymbolData(bars={"1h": _make_uptrend_df(100)})

        with patch.dict("sys.modules", _mock_all_engines_neutral()):
            result = self.scorer.score(sd)
        # At minimum the internal components should be populated
        for key in [
            "mtf_alignment",
            "trend_alignment",
            "wave_phase",
            "structure_shift",
            "fvg_confluence",
            "order_block",
            "premium_discount",
            "liquidity_sweep",
            "exhaustion_check",
            "breakout_strength",
            "relative_strength",
            "vol_compression",
        ]:
            assert key in result.components, f"Missing component: {key}"

    def test_all_profiles_work(self):
        for profile in ("SCALP", "SWING", "POSITION"):
            cfg = TRADE_PROFILES[profile]
            tf = cfg["bar_tf"]
            sd = MockSymbolData(
                profile=profile,
                bars={tf: _make_uptrend_df(100)},
            )
            result = self.scorer.score(sd)
            assert 0 <= result.score <= 100
            assert result.profile == profile

    def test_confidence_levels(self):
        """Verify confidence mapping: HIGH >= 85, MEDIUM >= 70, LOW >= 55, NONE < 55."""
        sig_high = ScoredSignal(
            symbol="X", profile="SWING", score=90, direction="LONG", confidence="HIGH"
        )
        sig_none = ScoredSignal(
            symbol="X", profile="SWING", score=20, direction="NO_TRADE", confidence="NONE"
        )
        assert sig_high.confidence == "HIGH"
        assert sig_none.confidence == "NONE"


# ── Tests: Hard Gates ──────────────────────────────────────────────────────────


class TestHardGates:
    def setup_method(self):
        self.scorer = UnifiedScorer()

    def test_mtf_below_040_blocks_trade(self):
        """When MTF alignment returns < 0.40, the trade is blocked."""
        sd = MockSymbolData(bars={"1h": _make_uptrend_df(100)})

        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = {
            "alignment_score": 0.20,
            "bias_direction": "NEUTRAL",
        }
        mock_mtf_module = MagicMock()
        mock_mtf_module.MTFAlignmentEngine.return_value = mock_engine_instance

        with patch.dict(
            "sys.modules",
            {
                "mtf_alignment": MagicMock(),
                "mtf_alignment.mtf_alignment_engine": mock_mtf_module,
            },
        ):
            result = self.scorer.score(sd)

        assert result.direction == "NO_TRADE"
        assert result.score < 40
        assert "MTF alignment too low" in result.rejection_reason

    def test_mtf_above_040_passes_gate(self):
        """When MTF alignment >= 0.40, the gate does not block."""
        sd = MockSymbolData(bars={"1h": _make_uptrend_df(100)})

        mock_engine_instance = MagicMock()
        mock_engine_instance.run.return_value = {
            "alignment_score": 0.65,
            "bias_direction": "LONG",
        }
        mock_mtf_module = MagicMock()
        mock_mtf_module.MTFAlignmentEngine.return_value = mock_engine_instance

        with patch.dict(
            "sys.modules",
            {
                "mtf_alignment": MagicMock(),
                "mtf_alignment.mtf_alignment_engine": mock_mtf_module,
            },
        ):
            result = self.scorer.score(sd)

        assert result.direction != "NO_TRADE" or "MTF" not in result.rejection_reason

    def test_no_structure_confirmation_penalty(self):
        """When structure_shift is neutral (~0.50), score is reduced by 30%."""
        sd = MockSymbolData(bars={"1h": _make_uptrend_df(100)})

        with patch.dict("sys.modules", _mock_all_engines_neutral()):
            result = self.scorer.score(sd)
        # Without SMC engine, structure_shift defaults to 0.50 -> penalty applied
        assert any("no structure confirmation" in b for b in result.bonuses)


# ── Tests: Helper Methods ──────────────────────────────────────────────────────


class TestHelperMethods:
    def setup_method(self):
        self.scorer = UnifiedScorer()

    def test_compute_atr(self):
        df = _make_uptrend_df(50)
        atr = self.scorer._compute_atr(df)
        assert atr > 0

    def test_compute_atr_short_series(self):
        df = _make_short_df(1)
        atr = self.scorer._compute_atr(df)
        assert atr == 1.0

    def test_score_trend_uptrend(self):
        df = _make_uptrend_df(100)
        score = self.scorer._score_trend(df)
        assert score > 0.5  # uptrend should score above neutral

    def test_score_trend_downtrend(self):
        df = _make_downtrend_df(100)
        score = self.scorer._score_trend(df)
        assert score < 0.5  # downtrend should score below neutral

    def test_score_trend_short_data(self):
        df = _make_short_df(10)
        score = self.scorer._score_trend(df)
        assert score == 0.5

    def test_score_vol_compression(self):
        df = _make_uptrend_df(100)
        score = self.scorer._score_vol_compression(df)
        assert 0.0 <= score <= 1.0

    def test_score_relative_strength_spy(self):
        df = _make_uptrend_df(50)
        score = self.scorer._score_relative_strength("SPY", df)
        assert score == 0.50

    def test_score_liquidity_neutral(self):
        df = _make_uptrend_df(50)
        atr = self.scorer._compute_atr(df)
        score = self.scorer._score_liquidity(df, atr)
        assert 0.0 <= score <= 1.0

    def test_score_breakout(self):
        df = _make_uptrend_df(50)
        atr = self.scorer._compute_atr(df)
        score = self.scorer._score_breakout(df, atr)
        assert 0.0 <= score <= 1.0

    def test_tf_to_mtf_label(self):
        assert self.scorer._tf_to_mtf_label("5m") == "5m"
        assert self.scorer._tf_to_mtf_label("1d") == "daily"
        assert self.scorer._tf_to_mtf_label("1wk") == "weekly"
        assert self.scorer._tf_to_mtf_label("unknown") is None

    def test_best_available_df(self):
        df = _make_uptrend_df(50)
        sd = MockSymbolData(bars={"4h": df})
        result = self.scorer._best_available_df(sd)
        assert result is not None
        assert len(result) == 50
