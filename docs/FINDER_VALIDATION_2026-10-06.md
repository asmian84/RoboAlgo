# Does the Finder have an edge? Full-universe replay through the validation gate

**Date:** 2026-10-06 · **Status:** research result, not trading advice · **Reproduce:** commands at the bottom

## Verdict

**No. None of the Finder's presets clears the gate, and none comes close.** Replayed over every
ticker it could have scanned, the best portfolio it produces had an out-of-sample Sharpe of
**0.42**. Buying and holding SPY over the same window scored
**1.29**. After declaring the 13 configurations tested, a strategy would
need an out-of-sample Sharpe near **2.0** to be statistically credible; one pre-registered test would
need about **1.0**.

This is a statement about *these rules with these exits*: enter on a break of the compressed range,
stop at its low, target one range-height up. It is not a statement that no screen can work.

## What was scanned: the entire group

| | |
|---|---:|
| US stocks listed on the download date | 11,937 |
| Preferred shares (`JPM/PK` etc.) that cannot be fetched | 361 |
| Symbols with price history returned | 11,518 |
| **Tickers scanned with the full rule** | **10,631** |
| Tickers too new for the engine (under 282 bars) | 887 |
| **Ticker-days evaluated (whole-universe run)** | **19,996,081** |
| Tracked large-cap/ETF instruments scanned (2010 onward) | 234 |
| Tickers that ever produced a signal: `signal` / `pennyking` / `maker` / `penny_floor1` | 165 / 687 / 1,435 / 526 |

Nothing was sampled or hand-picked. Every ticker was evaluated on every bar. The `pennyking` and
`maker` presets ran on the whole listed universe; the `signal` preset ran on the 234 instruments the
database tracks, because that is the universe the live `/scan` endpoint reads.

## Results (out of sample, 2024-01-01 to 2026-10-06; 693 trading days)

### Portfolio A, "top 10": at most 10 positions, filled in the Finder's own ranking. This is what could be traded.

| Preset | Universe | OOS Sharpe | OOS max drawdown | OOS hit rate | OOS trades | In-sample Sharpe | Deflated Sharpe (needs 0.95) | Gate verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `signal` | Large caps + ETFs (234 tracked) | 0.42 | 14.3% | 0.66 | 616 | 0.63 | 0.171 | **REJECTED** |
| `pennyking` | Sub-$5, all listed | 0.20 | 51.8% | 0.59 | 722 | 0.25 | 0.069 | **REJECTED** |
| `maker` | $5-20, all listed | -0.77 | 40.6% | 0.56 | 866 | -0.19 | 0.002 | **REJECTED** |
| `penny_floor1` | Sub-$5 with a $1 floor | -0.89 | 59.3% | 0.57 | 688 | 0.35 | 0.001 | **REJECTED** |

### Portfolio B, "uncapped": every signal taken, equal-weighted. This measures the screen's edge with no selection.

| Preset | Universe | OOS Sharpe | OOS max drawdown | OOS hit rate | OOS trades | In-sample Sharpe | Deflated Sharpe (needs 0.95) | Gate verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `signal` | Large caps + ETFs (234 tracked) | 0.09 | 31.2% | 0.66 | 1,160 | 0.37 | 0.204 | **REJECTED** |
| `pennyking` | Sub-$5, all listed | 0.27 | 69.7% | 0.60 | 1,503 | 0.14 | 0.283 | **REJECTED** |
| `maker` | $5-20, all listed | -0.25 | 42.9% | 0.58 | 4,804 | -0.07 | 0.081 | **REJECTED** |
| `penny_floor1` | Sub-$5 with a $1 floor | -0.91 | 67.9% | 0.59 | 1,347 | 0.25 | 0.006 | **REJECTED** |

**Buy-and-hold SPY, same windows:** in-sample Sharpe 0.71 (2019-03 to 2023), out-of-sample
**1.29** with a 18.8% drawdown. Every Finder variant is below it.

Gate limits: OOS Sharpe >= 1.0, max drawdown <= 15%, hit rate >= 55%, >= 30 trades, deflated Sharpe >= 0.95.
Every candidate failed on Sharpe and on deflated Sharpe. All but `signal` (top-10) also failed on drawdown.

## Why it fails

All figures in this section are over each preset's whole replay (2019-2026 for the listed universe, 2012-2026 for
`signal`), gross of costs, so they describe the rule and not one window.

**1. The reward-to-risk the Finder displays is not the one a trader gets.** Every row shows "R:R 1.0:1",
measured from the entry level. But **55-57% of fills happen above that entry level** (the stock has already gapped
or run past the trigger when the order works), and 39-50% of them by more than 1%. Filled higher, the stop is
further away and the target is nearer:

| Preset | Fills above displayed entry | Planned R:R | Realised R:R at the fill (median / mean) | Average win | Average loss | Win / loss size |
|---|---:|---:|---:|---:|---:|---:|
| `signal` | 57.2% | 1.00 | 0.87 / 0.68 | +3.27% | -5.23% | 0.63 |
| `pennyking` | 54.9% | 1.00 | 0.85 / 0.68 | +9.98% | -15.84% | 0.63 |
| `maker` | 56.6% | 1.00 | 0.86 / 0.68 | +6.31% | -9.76% | 0.65 |

**2. A high hit rate is not an edge, and here it only just clears break-even.** With wins about 0.63x the size of
losses, the break-even hit rate is about 61%. The actual hit rates are 66.5% (`signal`), 65.1% (`pennyking`) and
62.6% (`maker`): above break-even by 2 to 5 points, which is a thin gross edge of roughly +0.1% to +1.0% per trade.
Every preset clears the gate's 55% hit-rate rule, and that rule says nothing about whether the money is made.

**3. Gaps are a real but secondary cost.** Gap-through-stop exits (the stock opens below the stop, so the exit is
the open) are only 3-5% of trades but account for **11-18% of all losing-trade losses**, and a gap exit costs about
1.3x a clean stop (sub-$5: mean -20.3% against a clean stop's median -14.6%). They matter, but they do not
explain the result.

**4. Costs erase what is left.**

| Preset | Gross per trade | Assumed round-trip cost | Net per trade | Profit factor |
|---|---:|---:|---:|---:|
| `signal` | +0.34% | 0.10% | +0.24% | 1.12 |
| `pennyking` | +0.12% | 1.00% | -0.88% | 0.84 |
| `maker` | -0.12% | 0.40% | -0.52% | 0.85 |
| `penny_floor1` | -0.07% | 1.00% | -1.07% | 0.81 |

`pennyking` earns +0.12% gross per trade against an assumed 1.00% round trip; `maker` is negative before costs.
The cost levels are assumptions (0.10% large caps, 0.40% mid, 1.00% sub-$5), not measurements. The sub-$5
number is the most uncertain and the one that matters most: if real round-trip costs there were 0.1%, the
gross edge is still only about 0.1%.

**5. It is not stable.** `signal` top-10 yearly Sharpe: 2012: +0.75, 2013: +2.53, 2014: +0.54, 2015: -0.09, 2016: -0.32, 2017: +3.00, 2018: -0.61, 2019: +1.48, 2020: +0.66, 2021: +1.27, 2022: -1.93, 2023: -0.31, 2024: +0.75, 2025: +0.77, 2026: -0.70.
Positive in 9 of 15 years; the three worst are 2022, 2026 to date and 2018.

## The result does not depend on my fill conventions

The default replay is deliberately pessimistic: a bar that could hit both stop and target is a stop, and a
bar that fills mid-range is assumed to have made its low first. The opposite, optimistic convention, which is a
bound and not a candidate, gives:

| Preset | Default (pessimistic) top-10 OOS Sharpe | Optimistic bound top-10 OOS Sharpe | Default uncapped | Optimistic uncapped |
|---|---:|---:|---:|---:|
| `signal` | 0.42 | 0.60 | 0.09 | 0.17 |
| `pennyking` | 0.20 | 0.20 | 0.27 | 0.30 |
| `maker` | -0.77 | -0.71 | -0.25 | -0.15 |
| `penny_floor1` | -0.89 | -0.90 | -0.91 | -0.81 |

Best case Sharpe is 0.60, still below the 1.0 limit and SPY's 1.29.

## How the trials were counted

13 configurations were evaluated and every candidate was deflated against all 13: three presets x
(base, hold 20 bars, hold 60 bars, "only while still below entry") plus a $1-floor variant of the sub-$5
preset. The optimistic-bound runs above are a sensitivity check and are excluded from the ledger, because
none of them is a candidate. Preset thresholds were set by judgement from distributions of recent data, not by
searching for profit. That is not the same as being independent of that data, so the out-of-sample window is
not perfectly clean.

## Biases and limits, all of which flatter the result

* **Survivorship.** The universe is what is listed *today*. Delisted, bankrupt and acquired companies are
  absent, and penny stocks delist constantly. Real performance would be **worse**. A strategy that fails
  here fails for real; one that passed here would still not be proven.
* **The out-of-sample window is a strong bull market** (SPY Sharpe 1.29). Long-only trend setups had the best
  conditions they are likely to get, and still lagged the index.
* **Reverse splits.** Yahoo split-adjusts history retroactively, so a stock that traded at $0.15 and later
  reverse-split 1:20 shows as $3.00 in the past. Price-band and volume filters then see wrong historical values.
* **One bias runs the other way.** The price series is not dividend-adjusted, so a large special or liquidating
  distribution (the worst trade in the `maker` group, ELME at -84.5%, is one) appears as a loss the holder
  was actually paid for. It is rare.

## What this does and does not show

It shows that the Finder, as built, picks setups with no demonstrable edge over buying the index, and that its
displayed 1:1 reward-to-risk is not what a trader experiences once fills above the entry level, gaps and
costs are included.

It does **not** show that the underlying ideas (compression, trend, breakout) are worthless, that other exits
or a regime filter could not help, or anything about the short side, which the engine does not have. Those are
testable now, and each one tested must be added to the trial count.

## Since this report: the displayed R:R is fixed

The Finder now shows each setup's entry status and the reward-to-risk from the price a buyer would actually pay
(`entry_status`, `rr_real`, `risk_pct` from `scanner_engine.finder.trade_levels`). On the 2026-10-06 large-cap scan only
6 of 13 hits were takeable as displayed. This does not change the verdict above: it makes the displayed number honest,
it does not add an edge.

## What you now have

This harness is reusable: any rule expressed as signals plus an exit model can be replayed over the full
universe and put through the gate in seconds. Variants worth pre-registering before running (each adds to
the trial count, and the deflated-Sharpe bar rises with it): a market-regime filter, trailing exits,
position sizing by risk, an actual short-side rule.

## Reproduce

```bash
python scripts/download_universe_history.py     # ~5 min, writes ~/.roboalgo/history (242 MB)
python scripts/validate_finder.py               # replay: ~15 s on 16 cores
python scripts/gate_finder.py                   # the gate verdicts
python scripts/finder_sensitivity.py            # optimistic bound, SPY, scorecard, coverage
python -m pytest tests/test_finder_backtest -q  # 28 tests incl. parity with the production engines
```

Verification of the replay itself: features match what the production engines stored, exactly (moving averages and
range bounds identical, `is_compressed` 100% agreement, compression score within 0.01) on 8 instruments x 1,500 bars; a
feature at day *t* is identical whether or not later bars exist; seven deliberate bugs (target-before-stop, wrong gap
fills, two look-ahead leaks, a removed one-order-per-setup guard, an ignored slot cap) were each caught by the tests.
