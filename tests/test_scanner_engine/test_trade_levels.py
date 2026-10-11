"""Entry status and real R:R in scanner_engine.finder.trade_levels.

The Finder prints "R:R 1.0:1" on every row, but that is measured from the entry
level. Replaying it over the whole listed universe showed 55-57% of fills happen
ABOVE the displayed entry, so the realised ratio averaged 0.68. These tests pin
the five states and the real ratio so the display cannot drift back to the
flattering number.
"""
import numpy as np
import pytest

from scanner_engine.finder import AT_ENTRY_PCT, ENTRY_STATUSES, trade_levels

# range 90..100  ->  entry 100, stop 90, target 110 (a 10-point height)
RH, RL = 100.0, 90.0
E, S, T = 100.0, 90.0, 110.0


def lv(price):
    return trade_levels(RH, RL, price)


def test_levels_are_unchanged_by_the_new_fields():
    t = lv(95.0)
    assert (t["entry"], t["stop"], t["target"], t["risk_reward"]) == (E, S, T, 1.0)
    assert t["distance_to_entry_pct"] == round((E - 95.0) / 95.0 * 100, 2)   # + means still below the trigger


def test_below_entry_is_judged_from_the_entry_level_not_the_price():
    t = lv(95.0)
    assert t["entry_status"] == "below_entry"
    assert t["fill_ref"] == E                  # a buy-stop fills at the break, not at today's price
    assert t["rr_real"] == 1.0
    assert t["risk_pct"] == pytest.approx(10.0) and t["reward_pct"] == pytest.approx(10.0)


@pytest.mark.parametrize("price", [99.6, 100.0, 100.4])
def test_at_entry_within_the_tolerance_on_either_side(price):
    assert lv(price)["entry_status"] == "at_entry"


def test_the_tolerance_boundary_is_exact():
    inside = E * (1 + AT_ENTRY_PCT / 100) - 1e-6
    beyond = E * (1 + AT_ENTRY_PCT / 100) + 1e-3
    assert lv(inside)["entry_status"] == "at_entry"
    assert lv(beyond)["entry_status"] == "past_entry"
    assert lv(E * (1 - AT_ENTRY_PCT / 100) + 1e-6)["entry_status"] == "at_entry"
    assert lv(E * (1 - AT_ENTRY_PCT / 100) - 1e-3)["entry_status"] == "below_entry"


def test_past_entry_uses_the_current_price_and_the_ratio_gets_worse():
    t = lv(105.0)
    assert t["entry_status"] == "past_entry"
    assert t["fill_ref"] == 105.0
    assert t["rr_real"] == pytest.approx((T - 105.0) / (105.0 - S), abs=0.01)     # 5 / 15
    assert t["rr_real"] < t["risk_reward"]
    assert t["risk_pct"] == pytest.approx((105 - 90) / 105 * 100, abs=0.01)
    assert t["reward_pct"] == pytest.approx((110 - 105) / 105 * 100, abs=0.01)


def test_the_further_past_entry_the_worse_the_ratio():
    ratios = [lv(p)["rr_real"] for p in (101.0, 103.0, 106.0, 109.0)]
    assert ratios == sorted(ratios, reverse=True) and len(set(ratios)) == 4


@pytest.mark.parametrize("price,status", [(90.0, "below_stop"), (80.0, "below_stop"),
                                          (110.0, "past_target"), (125.0, "past_target")])
def test_dead_setups_have_no_ratio_and_no_fill_reference(price, status):
    t = lv(price)
    assert t["entry_status"] == status
    assert t["rr_real"] is None and t["fill_ref"] is None and t["risk_pct"] is None and t["reward_pct"] is None


def test_one_tick_inside_either_edge_is_still_a_live_setup():
    assert lv(90.01)["entry_status"] == "below_entry"
    assert lv(109.99)["entry_status"] == "past_entry"
    assert lv(109.99)["rr_real"] is not None


def test_no_levels_means_no_status_and_a_stable_shape():
    for args in ((None, None, 100.0), (90.0, 100.0, 95.0), (100.0, 100.0, 95.0), ("x", 1, 1)):
        t = trade_levels(*args)
        assert t["entry"] is None and t["entry_status"] is None and t["rr_real"] is None
    assert set(trade_levels(None, None, 1.0)) == set(lv(95.0)) - {"distance_to_entry_pct"}


def test_no_price_still_returns_the_levels_but_no_status():
    t = trade_levels(RH, RL, None)
    assert t["entry"] == E and t["entry_status"] is None and t["rr_real"] is None


def test_invariant_the_real_ratio_never_beats_the_planned_one():
    # Entry-to-stop and entry-to-target are the same height, and the fill reference is never below
    # entry, so realised reward-to-risk can only be equal to or worse than the planned 1.0.
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(4000):
        low = rng.uniform(0.5, 500)
        height = low * rng.uniform(0.01, 0.4)
        price = low * rng.uniform(0.7, 1.9)
        t = trade_levels(low + height, low, price)
        assert t["entry_status"] in ENTRY_STATUSES
        seen.add(t["entry_status"])
        if t["entry_status"] in ("past_target", "below_stop"):
            assert t["rr_real"] is None
        else:
            assert 0 <= t["rr_real"] <= t["risk_reward"] + 0.011      # a hair under target rounds to 0.00
            assert t["fill_ref"] >= t["entry"]
    assert seen == set(ENTRY_STATUSES), "the random sweep should reach every state"
