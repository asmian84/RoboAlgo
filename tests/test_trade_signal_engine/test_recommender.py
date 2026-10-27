"""Tests for TradeRecommender — SMC sniper entry and position sizing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from trade_signal_engine.recommender import (
    TRADE_PROFILES,
    TradeRecommendation,
    TradeRecommender,
)
from trade_signal_engine.scorer import ScoredSignal


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_signal(
    symbol: str = "AAPL",
    profile: str = "SWING",
    score: float = 80.0,
    direction: str = "LONG",
    confidence: str = "HIGH",
    components: dict | None = None,
    component_details: dict | None = None,
    rejection_reason: str = "",
) -> ScoredSignal:
    return ScoredSignal(
        symbol=symbol,
        profile=profile,
        score=score,
        direction=direction,
        confidence=confidence,
        components=components or {},
        component_details=component_details or {},
        rejection_reason=rejection_reason,
        bonuses=[],
    )


CURRENT_PRICE = 150.0
ATR = 3.0
ACCOUNT = 10_000.0


# ── Tests: TradeRecommendation dataclass fields ───────────────────────────────


class TestTradeRecommendationFields:
    def test_required_fields_present(self):
        """TradeRecommendation has all required fields with correct defaults."""
        rec = TradeRecommendation(
            symbol="TSLA",
            profile="SWING",
            direction="LONG",
            verdict="BUY",
            score=85.0,
            confidence="HIGH",
        )
        assert rec.symbol == "TSLA"
        assert rec.profile == "SWING"
        assert rec.direction == "LONG"
        assert rec.verdict == "BUY"
        assert rec.score == 85.0
        assert rec.confidence == "HIGH"
        assert rec.entry == 0.0
        assert rec.stop == 0.0
        assert rec.target == 0.0
        assert rec.risk_reward == 0.0
        assert rec.level_source == ""
        assert rec.account_size == 10_000.0
        assert rec.risk_amount == 0.0
        assert rec.position_size == 0
        assert rec.position_value == 0.0
        assert rec.projected_profit == 0.0
        assert rec.projected_profit_pct == 0.0
        assert rec.rejection_reason == ""
        assert rec.entry_notes == []
        assert rec.component_summary == {}

    def test_mutable_defaults_are_independent(self):
        """Mutable list/dict defaults do not share state between instances."""
        r1 = TradeRecommendation(symbol="A", profile="SWING", direction="LONG", verdict="BUY", score=75, confidence="HIGH")
        r2 = TradeRecommendation(symbol="B", profile="SWING", direction="LONG", verdict="BUY", score=75, confidence="HIGH")
        r1.entry_notes.append("note")
        assert r2.entry_notes == []


# ── Tests: NO_TRADE signal ─────────────────────────────────────────────────────


class TestNoTradeSignal:
    def test_no_trade_direction_returns_no_trade_verdict(self):
        signal = _make_signal(direction="NO_TRADE", confidence="NONE", score=20.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.verdict == "NO_TRADE"

    def test_no_trade_uses_rejection_reason(self):
        signal = _make_signal(
            direction="NO_TRADE",
            confidence="NONE",
            score=20.0,
            rejection_reason="MTF alignment too low",
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.rejection_reason == "MTF alignment too low"

    def test_no_trade_fallback_rejection_reason(self):
        """When rejection_reason is empty, a default message is set."""
        signal = _make_signal(direction="NO_TRADE", confidence="NONE", score=20.0, rejection_reason="")
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.rejection_reason != ""

    def test_no_current_price_returns_no_trade(self):
        """If current_price=0 and direction is not NO_TRADE, still returns NO_TRADE."""
        signal = _make_signal(direction="LONG", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=0.0)
        assert rec.verdict == "NO_TRADE"
        assert "price" in rec.rejection_reason.lower()


# ── Tests: LONG HIGH confidence → BUY ────────────────────────────────────────


class TestLongHighConfidence:
    def setup_method(self):
        self.signal = _make_signal(direction="LONG", confidence="HIGH", score=88.0)
        self.rec = TradeRecommender().recommend(
            self.signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )

    def test_verdict_is_buy(self):
        assert self.rec.verdict == "BUY"

    def test_entry_is_positive(self):
        assert self.rec.entry > 0

    def test_stop_below_entry_for_long(self):
        assert self.rec.stop < self.rec.entry

    def test_target_above_entry_for_long(self):
        assert self.rec.target > self.rec.entry

    def test_direction_preserved(self):
        assert self.rec.direction == "LONG"

    def test_uses_atr_fallback(self):
        assert self.rec.level_source == "atr_fallback"

    def test_entry_notes_populated(self):
        assert len(self.rec.entry_notes) > 0


# ── Tests: SHORT HIGH confidence → SELL ──────────────────────────────────────


class TestShortHighConfidence:
    def setup_method(self):
        self.signal = _make_signal(direction="SHORT", confidence="HIGH", score=82.0)
        self.rec = TradeRecommender().recommend(
            self.signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )

    def test_verdict_is_sell(self):
        assert self.rec.verdict == "SELL"

    def test_entry_is_positive(self):
        assert self.rec.entry > 0

    def test_stop_above_entry_for_short(self):
        assert self.rec.stop > self.rec.entry

    def test_target_below_entry_for_short(self):
        assert self.rec.target < self.rec.entry

    def test_direction_preserved(self):
        assert self.rec.direction == "SHORT"

    def test_uses_atr_fallback(self):
        assert self.rec.level_source == "atr_fallback"


# ── Tests: MEDIUM confidence ──────────────────────────────────────────────────


class TestMediumConfidence:
    def test_long_medium_is_buy(self):
        signal = _make_signal(direction="LONG", confidence="MEDIUM", score=72.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.verdict == "BUY"

    def test_short_medium_is_sell(self):
        signal = _make_signal(direction="SHORT", confidence="MEDIUM", score=71.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.verdict == "SELL"


# ── Tests: LOW confidence → HOLD ──────────────────────────────────────────────


class TestLowConfidence:
    def test_low_confidence_still_buys_above_the_score_threshold(self):
        """Confidence no longer decides the verdict: the score does.

        The recommender used to downgrade LOW confidence to HOLD. It now keys BUY and
        SELL off score >= 55, behind the minimum-gain gate, so a LOW-confidence signal
        scoring 58 is still actionable.
        """
        signal = _make_signal(direction="LONG", confidence="LOW", score=58.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.verdict == "BUY"

    def test_score_below_the_threshold_is_hold(self):
        """Between 35 and 55 the signal is watchlist only, whatever the confidence says."""
        signal = _make_signal(direction="LONG", confidence="HIGH", score=45.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.verdict == "HOLD"


# ── Tests: Position sizing ────────────────────────────────────────────────────


class TestPositionSizing:
    def setup_method(self):
        self.signal = _make_signal(direction="LONG", confidence="HIGH", score=85.0)
        self.rec = TradeRecommender().recommend(
            self.signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )

    def test_risk_amount_positive(self):
        assert self.rec.risk_amount > 0

    def test_position_size_positive(self):
        assert self.rec.position_size > 0

    def test_position_value_positive(self):
        assert self.rec.position_value > 0

    def test_risk_amount_within_profile_limit(self):
        """Risk amount should not exceed profile max_risk_pct of account."""
        profile_cfg = TRADE_PROFILES["SWING"]
        max_risk = ACCOUNT * (profile_cfg["max_risk_pct"] / 100)
        assert self.rec.risk_amount <= max_risk + 0.01  # small float tolerance

    def test_position_value_equals_size_times_entry(self):
        expected = round(self.rec.position_size * self.rec.entry, 2)
        assert self.rec.position_value == expected

    def test_position_sizing_respects_account_size(self):
        """Larger account should produce proportionally larger position sizes."""
        big_rec = TradeRecommender().recommend(
            self.signal, atr=ATR, current_price=CURRENT_PRICE, account_size=100_000.0
        )
        assert big_rec.position_size > self.rec.position_size

    def test_atr_estimated_when_zero(self):
        """When atr=0 is passed, 2% of current_price is used as estimate."""
        rec = TradeRecommender().recommend(
            self.signal, atr=0, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )
        # ATR estimate = 150 * 0.02 = 3.0 → same as ATR=3.0 test
        assert rec.position_size > 0
        assert rec.entry > 0


# ── Tests: Profit projection ──────────────────────────────────────────────────


class TestProfitProjection:
    def test_projected_profit_positive_for_long(self):
        signal = _make_signal(direction="LONG", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(
            signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )
        assert rec.projected_profit > 0

    def test_projected_profit_positive_for_short(self):
        signal = _make_signal(direction="SHORT", confidence="HIGH", score=82.0)
        rec = TradeRecommender().recommend(
            signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )
        assert rec.projected_profit > 0

    def test_projected_profit_pct_is_the_move_from_entry_to_target(self):
        """projected_profit_pct is the price move to target, not profit over the account.

        The two coincide only when the position absorbs the whole account, so the old
        (profit / account) assertion broke once sizing became risk-capped. The minimum
        gain gate reads this field as a percentage move, which fixes its meaning.
        """
        signal = _make_signal(direction="LONG", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(
            signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT
        )
        assert rec.projected_profit_pct == round((rec.target - rec.entry) / rec.entry * 100, 2)
        # the dollar figure stays the per-share move times the size
        assert rec.projected_profit == pytest.approx(
            round(rec.position_size * (rec.target - rec.entry), 2)
        )

    def test_higher_rr_profile_higher_profit(self):
        """POSITION profile (4:1 R:R) should produce more projected profit than SCALP (2.5:1)."""
        scalp_signal = _make_signal(profile="SCALP", direction="LONG", confidence="HIGH", score=85.0)
        pos_signal = _make_signal(profile="POSITION", direction="LONG", confidence="HIGH", score=85.0)
        scalp_rec = TradeRecommender().recommend(scalp_signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT)
        pos_rec = TradeRecommender().recommend(pos_signal, atr=ATR, current_price=CURRENT_PRICE, account_size=ACCOUNT)
        assert pos_rec.projected_profit > scalp_rec.projected_profit


# ── Tests: SMC sniper entry priority ──────────────────────────────────────────


class TestSmcSniperEntry:
    def _signal_with_smc(self, direction: str = "LONG") -> ScoredSignal:
        sniper = 149.0  # within 5% of CURRENT_PRICE=150
        zone_bottom = 148.0
        zone_top = 150.0
        return _make_signal(
            direction=direction,
            confidence="HIGH",
            score=90.0,
            component_details={
                "smc": {
                    "sniper_entry": sniper,
                    "entry_zone_bottom": zone_bottom,
                    "entry_zone_top": zone_top,
                    "zone": "DISCOUNT",
                    "in_ote": True,
                }
            },
        )

    def test_smc_sniper_used_as_entry_long(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("LONG"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.level_source == "smc_sniper"
        assert rec.entry == 149.0

    def test_smc_sniper_stop_below_entry_long(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("LONG"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.stop < rec.entry

    def test_smc_sniper_target_above_entry_long(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("LONG"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.target > rec.entry

    def test_smc_sniper_stop_above_entry_short(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("SHORT"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.stop > rec.entry

    def test_smc_sniper_target_below_entry_short(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("SHORT"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.target < rec.entry

    def test_smc_ote_note_added(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("LONG"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert any("OTE" in note for note in rec.entry_notes)

    def test_smc_zone_note_added(self):
        rec = TradeRecommender().recommend(
            self._signal_with_smc("LONG"), atr=ATR, current_price=CURRENT_PRICE
        )
        assert any("DISCOUNT" in note for note in rec.entry_notes)

    def test_smc_sniper_ignored_when_too_far(self):
        """Sniper entry >5% away from current price should fall through to ATR fallback."""
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            component_details={
                "smc": {
                    "sniper_entry": 100.0,  # >5% from 150
                    "entry_zone_bottom": 99.0,
                    "entry_zone_top": 101.0,
                    "zone": "DISCOUNT",
                    "in_ote": False,
                }
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.level_source == "atr_fallback"


# ── Tests: Level pounce priority ──────────────────────────────────────────────


class TestLevelPounceEntry:
    def _signal_with_pounce(self) -> ScoredSignal:
        return _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            component_details={
                "levels": {
                    "pounce_ideas": [
                        {"entry": 148.5, "stop": 146.0, "target": 157.0}
                    ]
                }
            },
        )

    def test_pounce_entry_used(self):
        rec = TradeRecommender().recommend(
            self._signal_with_pounce(), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.level_source == "level_pounce"
        assert rec.entry == 148.5

    def test_pounce_stop_used_and_target_stretched_to_the_minimum(self):
        """The pounce stop is taken as given; its target is raised to the profile's floor.

        148.50 to 157.00 is a 5.7% move, under SWING's 10% minimum target, so the
        recommender stretches the target to 163.35 and records why in the notes.
        """
        rec = TradeRecommender().recommend(
            self._signal_with_pounce(), atr=ATR, current_price=CURRENT_PRICE
        )
        assert rec.stop == 146.0
        assert rec.target == pytest.approx(163.35)
        assert any("stretched" in note for note in rec.entry_notes)

    def test_pounce_target_beyond_the_minimum_is_kept(self):
        """A pounce target already past the floor is used unchanged."""
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            component_details={
                "levels": {
                    "pounce_ideas": [
                        {"entry": 148.5, "stop": 146.0, "target": 175.0}
                    ]
                }
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.target == 175.0
        assert not any("stretched" in note for note in rec.entry_notes)

    def test_pounce_note_added(self):
        rec = TradeRecommender().recommend(
            self._signal_with_pounce(), atr=ATR, current_price=CURRENT_PRICE
        )
        assert any("pounce" in note.lower() for note in rec.entry_notes)


# ── Tests: ATR fallback ────────────────────────────────────────────────────────


class TestAtrFallback:
    def test_atr_fallback_used_when_no_smc_no_pounce(self):
        signal = _make_signal(direction="LONG", confidence="HIGH", score=80.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.level_source == "atr_fallback"

    def test_atr_fallback_entry_is_current_price(self):
        signal = _make_signal(direction="LONG", confidence="HIGH", score=80.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.entry == round(CURRENT_PRICE, 2)

    def test_atr_fallback_stop_distance_matches_profile(self):
        """SWING stop = current_price - ATR * 1.5."""
        signal = _make_signal(profile="SWING", direction="LONG", confidence="HIGH", score=80.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        expected_stop = round(CURRENT_PRICE - ATR * TRADE_PROFILES["SWING"]["stop_atr_mult"], 2)
        assert rec.stop == expected_stop

    def test_atr_fallback_note_contains_profile_info(self):
        signal = _make_signal(profile="SWING", direction="LONG", confidence="HIGH", score=80.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert any("ATR" in note for note in rec.entry_notes)


# ── Tests: Risk:Reward ratio ───────────────────────────────────────────────────


class TestRiskRewardRatio:
    def test_rr_matches_profile_target_rr_long(self):
        """ATR fallback R:R should match the profile's target_rr."""
        profile = "SWING"
        signal = _make_signal(profile=profile, direction="LONG", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        expected_rr = TRADE_PROFILES[profile]["target_rr"]
        # Allow ±0.1 tolerance due to rounding
        assert abs(rec.risk_reward - expected_rr) < 0.2

    def test_rr_matches_profile_target_rr_short(self):
        profile = "SWING"
        signal = _make_signal(profile=profile, direction="SHORT", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        expected_rr = TRADE_PROFILES[profile]["target_rr"]
        assert abs(rec.risk_reward - expected_rr) < 0.2

    def test_rr_positive_for_valid_trade(self):
        signal = _make_signal(direction="LONG", confidence="HIGH", score=85.0)
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.risk_reward > 0

    def test_scalp_rr_less_than_position_rr(self):
        """SCALP profile 2.5:1 < POSITION profile 4:1."""
        scalp_sig = _make_signal(profile="SCALP", direction="LONG", confidence="HIGH", score=85.0)
        pos_sig = _make_signal(profile="POSITION", direction="LONG", confidence="HIGH", score=85.0)
        scalp_rec = TradeRecommender().recommend(scalp_sig, atr=ATR, current_price=CURRENT_PRICE)
        pos_rec = TradeRecommender().recommend(pos_sig, atr=ATR, current_price=CURRENT_PRICE)
        assert scalp_rec.risk_reward < pos_rec.risk_reward

    def test_smc_rr_computed_correctly(self):
        """R:R = reward / risk using SMC zone-derived stop/target."""
        sniper = 149.0
        zone_bottom = 148.0  # stop = 148 - 3*0.3 = 147.1
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=90.0,
            component_details={
                "smc": {
                    "sniper_entry": sniper,
                    "entry_zone_bottom": zone_bottom,
                    "entry_zone_top": 150.0,
                    "zone": "DISCOUNT",
                    "in_ote": False,
                }
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        risk = rec.entry - rec.stop
        reward = rec.target - rec.entry
        expected_rr = round(reward / risk, 1)
        assert rec.risk_reward == expected_rr


# ── Tests: Component summary ──────────────────────────────────────────────────


class TestComponentSummary:
    def test_component_summary_populated(self):
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            components={
                "mtf_alignment": 0.80,   # bullish
                "trend_alignment": 0.20, # bearish
                "wave_phase": 0.50,      # neutral
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert "top_bullish" in rec.component_summary
        assert "top_bearish" in rec.component_summary
        assert "hard_gates_passed" in rec.component_summary
        assert rec.component_summary["hard_gates_passed"] is True

    def test_bullish_components_captured(self):
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            components={"mtf_alignment": 0.80},
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert any("mtf_alignment" in item for item in rec.component_summary["top_bullish"])

    def test_bearish_components_captured(self):
        signal = _make_signal(
            direction="SHORT",
            confidence="HIGH",
            score=82.0,
            components={"trend_alignment": 0.20},
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert any("trend_alignment" in item for item in rec.component_summary["top_bearish"])

    def test_neutral_components_not_in_summary(self):
        """Components at 0.50 should not appear in top_bullish or top_bearish."""
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=85.0,
            components={"wave_phase": 0.50},
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        all_items = rec.component_summary["top_bullish"] + rec.component_summary["top_bearish"]
        assert not any("wave_phase" in item for item in all_items)


# ── Tests: SMC priority over Level pounce ────────────────────────────────────


class TestEntryPriority:
    def test_smc_takes_priority_over_pounce(self):
        """When both SMC sniper and pounce ideas are present, SMC wins."""
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=90.0,
            component_details={
                "smc": {
                    "sniper_entry": 149.0,
                    "entry_zone_bottom": 148.0,
                    "entry_zone_top": 150.0,
                    "zone": "DISCOUNT",
                    "in_ote": False,
                },
                "levels": {
                    "pounce_ideas": [
                        {"entry": 147.0, "stop": 145.0, "target": 155.0}
                    ]
                },
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.level_source == "smc_sniper"

    def test_pounce_takes_priority_over_atr_fallback(self):
        """When only pounce is present (no SMC), pounce is used."""
        signal = _make_signal(
            direction="LONG",
            confidence="HIGH",
            score=80.0,
            component_details={
                "levels": {
                    "pounce_ideas": [
                        {"entry": 148.0, "stop": 146.0, "target": 156.0}
                    ]
                }
            },
        )
        rec = TradeRecommender().recommend(signal, atr=ATR, current_price=CURRENT_PRICE)
        assert rec.level_source == "level_pounce"
