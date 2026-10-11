# Learning the setups behind big runners: first round

**Date:** 2026-10-07 · **Status:** research result, not trading advice · **Registry:** `setup_engine/registry/registry.json`

> **Update, same day: round two and the delisted-data work are at the bottom ([jump](#round-two-and-the-delisted-data-work)).**
> The conclusion below, that no *runner* setup was found, still stands. What changed: three trend setups are now
> registered as approved for a narrower claim (they out-earn the average stock modestly), and scans tag hits with them.

## Verdict

**No setup was found that you can use.** The system learned 17 distinct patterns that beat chance on data
they were not learned from, with a placebo control showing the machinery does not invent patterns from
noise. All 17 are the same idea (a stock that has crashed 70%+ and is near its lows), and on inspection
none is usable: flagged stocks run 5x more often, but they also lose half their value 59-74% of the time
(28% for all stocks), and **their average return is no better than the average stock's**. AMD and NVDA were
flagged at the 2002 market bottom and again at the 2022 market low (both then ran), but **not** before AMD's
2015 run or NVDA's 2015-2019 runs. What the rules detect is a market-wide capitulation low, which is a
market-timing event, and at those dates they flag hundreds of stocks of which only a few are the next AMD.
The textbook setups (strong trend, tight base, golden cross) came out *below* chance for this goal.

None of those 17 is approved. The registry holds them all, with the reason each was rejected.

## What was built

A learning loop, not a list. `setup_engine/` has: scale-free features that use the same bar-count windows on
**daily, weekly and monthly** bars (so "the setup is the same on any timeframe" is tested, not assumed);
forward labels; a learner; robustness checks; an append-only trial ledger; and a registry that only scans
**approved** setups. Re-running `scripts/learn_setups.py` then `scripts/register_setups.py` is the evolution
step. The ledger count (241 attempts so far) only grows, so the multiple-testing correction stays honest.

| | |
|---|---|
| Universe | 6,521 liquid stocks (>= $500k/day), 714,631 observations, 2001-2026, from the whole listed universe |
| Label (fixed before any model ran) | reaches **5x within 3 years** (2.8% of observations; 1,580 stocks) |
| Splits | learn 2001-11, select 2015-17, judge **once** on 2021-23; each starts after the previous stage's 3-year labels end |
| Tests | 44 tests; every look-ahead and learning mutant I tried is caught, bar one equivalent case (a second guard already blocks it) |

## Why a naive version would have lied, and what the placebo found

Picking winners and describing their charts finds a "common setup" every time. Three guard rails matter:

1. **Lift against the same date's base rate.** The share of stocks reaching 5x swings from 0.2% (2007) to 7.2% (2020).
2. **Lift against the same stock's own base rate ("timing").** The first placebo run (outcomes circularly shifted
   in time, chart-to-outcome link severed, stock identity kept) validated **7 of 18** rules, all "which kind of
   stock" rules (deeply beaten-down, gap-prone). That is the survivorship-contaminated signal. The criterion was
   changed to timing lift **before any real result was seen**; after the change the placebo validated **0 of 14,
   0 of 19, 0 of 47**.
3. **Resample stocks, not observations.** Consecutive observations of one stock are not independent.

## Results

22 of 50 rules (47 learned + 3 textbook) validated on 2015-17 (17 distinct); all 22 held on the 2021-23 test era
with timing lift 1.18-1.67 and intervals above 1. Then two diagnostics, **added after the test result had been
seen** and therefore exploratory for this round (they are pre-registered for every later round):

| | flagged stocks | all stocks |
|---|---:|---:|
| Reach 5x in 3 years (2021-23) | 8.7-8.9% | 3.1% |
| Lose half or more in 3 years | **60-73%** | 28% |
| Median 3-year return | **-72% to -92%** | -10% |
| Mean 3-year return | +1% to +8% | +14% |
| Mean after trimming the top 1% of winners | **-34% to -13%** | +9% |

The same pattern holds in 2015-17: more 5x runners, mean return about equal to the market's, median -46% to -58%.

**Liquidity.** Pooled over stocks trading >= $10M/day, the timing lift is 1.06-1.59 and **8 of the 17 rules keep an
interval above 1**. Split into four narrower liquidity bands (checked for three rules) it is ~1.0-1.4 with wide
intervals that include 1. The evidence is fragile and ambiguous, not cleanly gone. The loss rate is what rejects
every one of them.

**Timeframe.** The rules learned independently from daily-only, weekly-only and monthly-only features all say
the same thing in different units (far below the 50-bar high, a deep fall over 30 bars, price far under the
10-bar average). That is mild support for your point that the shape is timeframe-invariant. It is the shape
of a crash, though, not of a base.

## AMD and NVDA

Across all 17 rules, AMD was flagged on 23 of 336 dates and NVDA on 21, clustered in market bottoms: 2001-03,
2008-09, 2011-13, and **2022-10-14**. Since 2014 each was flagged on that one date only.

* **2002-2003 lows.** In Oct 2002, 51-55% of every flagged stock reached 5x and none lost half. Every stock that
  had crashed rebounded off the market-wide bottom: a market-timing event, not a chart setup.
* **2022-10-14, the 2022 market low.** Both flagged, both ran: AMD reached 4.3x and NVDA 17.2x within three
  years. This is the best case for the idea, so here is the whole cohort. On that date 599 of 3,254 liquid stocks
  (18%) were flagged. 11.9% of them reached 5x (against 5.2% of all stocks), the median flagged stock **lost 31%**
  (all stocks: +11%), 44% lost half or more (all: 20%), and the mean was +69% (all: +53%). Of the 71 flagged
  stocks that reached 5x, 24 were liquid.
* **Not flagged:** AMD's 2015 low ($1.62), and NVDA's runs from 2015 to 2019. The rules had nothing to say about
  the lead-up to the runs that made them famous.
* Across all 6,323 flagged observations (1,514 stocks): 18.8% reached 5x, 7.0% reached 10x, 2.8% reached 20x,
  and **54% lost half or more**.

So the rules do pick up capitulation lows, and at one of them they found both stocks. They also found 597 others
that day. Telling the next NVDA from the rest needs something these rules do not have.

## Why this is hard to do with the data we have

**Survivorship.** The universe is stocks listed *today*. A stock that crashed 70% and went to zero was delisted
and is not in the data. So the crashed names that remain are, by construction, the ones that recovered. AMD and NVDA
are exactly those survivors, and the same is true of the examples that come to mind when you think "runner". For
every one there are many that did not, and the data cannot show them. Everything here is an upper bound.

**What would settle it:** a survivorship-free price dataset that includes delisted stocks (CRSP, Norgate,
Sharadar, EODHD, Polygon). With it the same procedure can run unchanged and give a real answer.

## What to do next, in order

1. Get delisted-stock history, then re-run `learn_setups.py`. It is the single largest source of error.
2. Change the objective. A 5x-in-3-years label rewards lottery tickets. Learning setups for a **risk-adjusted**
   outcome among **liquid** stocks (the ones AMD and NVDA actually were) asks a better question.
3. Keep the loop running quarterly: the ledger, the placebo and the robustness gate are the part that makes
   "evolve as a system" safe instead of self-deceiving.

## Reproduce

```bash
python scripts/download_universe_history.py --start 2000-01-01 --out-dir ~/.roboalgo/history_2000   # ~5 min
python scripts/build_setup_dataset.py          # features + labels, ~1 min
python scripts/learn_setups.py                 # learn, select, judge, plus 3 placebo runs, ~85 s
python scripts/register_setups.py              # robustness gate + registry
python -m pytest tests/test_setup_engine -q    # 44 tests
```


---

# Round two and the delisted-data work

## 1. Getting delisted data: what was possible

| Source | Delisted list | Delisted prices |
|---|---|---|
| Yahoo (yfinance) | no | **4 of 40 known delisted tickers returned anything, and those 4 were re-used symbols** (a different company now holds `BBBY`, `FRCB`, `SBNY`, `EMC`) |
| AlphaVantage (your key) | **yes: 9,550 rows (7,539 US stocks, 2,011 ETFs) with delisting dates**, saved to `~/.roboalgo/delisted/av_delisted.csv` | gated; the daily series is rate-limited / premium |
| FMP (your key) | yes | HTTP 402, paid tier |
| Finnhub (your key) | current symbols only | candles paywalled |
| Stooq, Nasdaq Data Link (anonymous) | n/a | a challenge page and HTTP 403 |

**No free source I can reach serves prices for delisted stocks**, and obtaining them means paying a vendor or opening
an account, which is your decision, not mine. The list does size the hole: delistings run 800-900 a year in 2021-23
(7,539 US stocks since the list begins), against ~11,900 survivors. The list is clearly incomplete before 2013 (40 for
2009), so it cannot size the early hole.

**What is ready for when you have a dataset** (Polygon, EODHD, Tiingo, Sharadar, Norgate and CRSP all cover delisted
stocks; I have not verified any vendor's current plan limits):

```bash
python scripts/ingest_delisted_history.py --source ~/data/delisted/        # any CSV/parquet, flexible column names
python scripts/build_setup_dataset.py --history ~/.roboalgo/history_all
python scripts/learn_setups.py --objective runner3y                         # then register_setups.py
```

The ingest does three things that matter. It **skips re-used tickers** instead of splicing two companies together.
It marks every ingested stock **terminal**, so the builder holds its last price flat through the forward window
(`setup_engine.features.hold_at_last_price`). Without that, observations whose window runs past a stock's final
bar would be dropped as "incomplete", which deletes exactly the observations carrying the damaging outcomes and
leaves the survivorship bias in the dataset built to remove it. And it never modifies the survivor cache. Holding the
last price is optimistic for a bankruptcy; a price series cannot hold the true terminal value.

## 2. Round two: a horizon where survivorship barely matters

Survivorship bias grows with the horizon. A stock missing from today's listings is one that later died, and a
**6-month** window rarely contains much of that slide, so the relationship between a setup and its 6-month outcome is
far less distorted than over three years. It is also the horizon your scans operate at. So the same procedure was rerun
on the data we have:

* **Label (fixed before any model ran):** doubles within 6 months. 4.0% of observations, 3,606 stocks. (+50% was 13%
  and varied from 4% to 37% by year, too common to mean "runner".)
* **Splits, purged to 6 months:** learn 2001-12, select 2013-08 to 2017, judge once on 2018-08 to 2026-03.
* **Criteria (pre-registered, and tighter than round one):** timing lift against the same stock's own base rate,
  Bonferroni-corrected, stable across volatility terciles, **and the flagged stocks must out-earn the same dates'
  average on winsorised returns**. The money requirement exists because round one's rules hit more often and earned no more.

**Result: 0 of 51 validated** (48 learned candidates + 3 textbook), and **0 validated on all three placebo seeds**.

| Why candidates failed selection (a rule can fail several) | |
|---|---:|
| did not out-earn the same dates' average | 49 |
| timing lift not significant after correcting for 51 trials | 29 |
| timing lift below 1.3 | 20 |
| lift negative inside a volatility tercile | 14 |
| fewer than 25 informative stocks | 3 |

The learned candidates are again "crashed 70%+" patterns with large lifts (2.2-2.9) and no excess return. The single
near-miss (+32.7% excess, interval above zero) rested on **23 observations from 21 stocks** and was rightly rejected.

## 3. The finding that is usable: trend setups earn a little more, and double less often

The three hand-written textbook setups (fixed in advance) were never expected to find runners, and they do not:
their timing lift for doubling is **below 1** (0.55-0.91). But measured by what matters for a screen, **average 6-month
return above the same dates' average, winsorised**, they are positive in every era:

| Setup | 2001-12 | 2013-17 | 2018-26 | Liquid stocks only (>= $10M/day) |
|---|---:|---:|---:|---:|
| Stage-2 trend | +1.3% | +0.9% [+0.2, +1.5] | +4.6% [+4.0, +5.2] | +2.7% [+2.1, +3.3] |
| Tight base breakout | +1.0% | +1.3% [+0.9, +1.7] | +4.2% [+3.8, +4.6] | +2.2% [+1.8, +2.7] |
| Compression + golden cross | +0.2% | +0.9% [+0.4, +1.4] | +2.3% [+1.8, +2.7] | +1.1% [+0.6, +1.6] |

That is the "grinding, consolidating, above the golden cross" picture you described at the start: it does **not** find
runners, and it does carry a small, consistent premium. This is the classic trend/momentum effect, and at this horizon
it is the part of the data that survivorship distorts least.

**How much to trust it.** Modest and provisional. It is a few percent per half-year *before costs*, against the equal-weighted
average of liquid listed stocks. The three rules select overlapping stocks, so this is one finding seen three ways, not three.
Two eras clear zero by their intervals, and the effect is about 1% per half-year until 2017 and 2-5% since. Their approval
is marked **provisional** in the registry: their excess return was first examined after the learned rules had failed.

## 4. What now happens in your scans

No UI change. The Finder's scan responses carry a new field, **`setup_evidence`**, on every result: which approved setups
that stock matches, with the measured excess return and its interval. It tags only. It never re-ranks, filters or drops a
result, and a result with no approved match gets an empty list.

* `scan()` (the tracked-instrument scan) tags from the stored price table; `scan_wide()` fetches two years of bars for just
  its final hits in one batched call and reports status in `stats.setup_evidence`.
* Relative-strength rules need a market-wide rank, so `setup_engine/registry/rs_reference.json` holds each momentum feature's
  distribution across 4,319 liquid stocks (refreshed whenever `build_setup_dataset.py` runs, flagged stale after 14 days).
* **Fidelity:** evaluated from two years of bars and the interpolated rank, a scanned stock matches the full-history, exact-rank
  definition on **364 of 364** sampled stocks (rank error 0.0004).
* **Failure is visible.** Too little history, a missing reference or a failed download gives `setup_evidence: null` plus
  `setup_evidence_error`, never an empty list that reads as "matches nothing".
* Live today: the large-cap/ETF scan tagged 14 of 17 hits; the $5-20 scan 36 of 52; the sub-$5 scan 9 of 26.

**What a tag does not mean.** It means "this stock is in a trend setup that has historically earned a few percent more than the
average stock over six months". It does not mean "will run", "will double" or "buy". The evidence covers liquid stocks
(>= $500k/day trading; the premium was also confirmed at >= $10M/day), and sub-$5 names were not tested as their own group. The
Finder's entry/stop/target levels are unchanged and still have no demonstrated trade edge.

## Reproduce

The research scripts read and write parquet, so they need `pip install pyarrow`. It is deliberately **not** in `requirements.txt`: the deployed app never touches parquet (the setup registry and rank reference are JSON), and the library is large. The ingest tests skip, with that reason, when it is absent.

```bash
python scripts/learn_setups.py --objective runner6m      # ~55 s, includes 3 placebo runs
python scripts/register_setups.py --objective runner6m   # robustness gate (excess mode) + registry
python -m pytest tests/test_setup_engine tests/test_ingest -q
```
