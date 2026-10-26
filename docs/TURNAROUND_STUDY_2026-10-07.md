# Does "monthly bottom, then a daily entry with a stop" work? A mechanical test

**Date:** 2026-10-07 · **Status:** research result, not trading advice · **Raw output:** `docs/turnaround_study_results.json` · **Trades:** `~/.roboalgo/turnaround/trades.parquet`

## Bottom line

**No demonstrated edge.** Over 25 years and 11,278 US stocks, the rule produced 1,568 trades. They averaged **+0.10R per trade after a 1% round-trip cost**, but that figure cannot be told apart from zero (95% interval -0.11R to +0.32R), and it does not hold up under four checks:

1. **It is carried by about ten trades.** The ten biggest winners sum to 153.5R, more than the whole 150.0R total. Without them the mean is -0.002R.
2. **It is carried by one year.** 2020 alone contributes +152.6R (173 trades, mean +0.88R). Without 2020 the mean is -0.002R.
3. **Random entries into the same kind of stock, on the same days, earn almost as much.** The matched control returns +0.05R; the difference is +0.05R, interval -0.08R to +0.18R. The volume breakout adds nothing beyond "being in a stock that has bottomed" (+0.01R).
4. **Survivorship is large enough to flip the sign.** Break-even is a missing-stock share of 7-9%; the lower-bound estimate of that share is 14-18%.

What the data does show is the **shape** of the payoff: about one trade in three wins; the median trade loses 0.7R; only 6.7% of trades reach +3R. "Boom, a winner" is the right-tail outcome, and it is rare.

## The rule, written as Finviz-style filters

```
BOTTOM   (read on the PREVIOUS bar)  2-Year High: 70% or more below
                                     Monthly Low: Higher low
TRIGGER  (on the signal bar)         20-Day High: New High
                                     Relative Volume: Over 2
LIQUIDITY                            Price: Over $1
                                     Dollar Volume 20d: Over 1M
ENTRY    next day's open             STOP  2x ATR(14) below entry
EXIT     3x ATR trailing stop (chandelier), or 120 bars, whichever is first
COST     1% round trip (also tested at 0.5% and 2%)
```

"Monthly bottom" is approximated without a calendar: the lowest low of the last 126 trading days must be higher than the lowest low of the 252 days before them. On your NAUT chart that is 0.84 (Aug 2026) against 0.62 (Sep 2025).

## How the test was kept honest

* **Fixed before any result:** the rule, the exit, the cost, the 12 configurations, the eras and the verdict rule are constants at the top of `scripts/study_turnaround.py`, and a pre-registration record was written to the append-only ledger before the run. The study was run once. The ledger went from 371 to 384 entries.
* **One change before any aggregate outcome:** the brief read BOTTOM on the same bar as the trigger. On your own chart that gives **zero signals**, because the breakout day lifts the price out of the "70% below" band (NAUT: 74.3% below the day before, 68.3% below at the breakout close). BOTTOM is therefore read on the previous bar, as a trader builds a watchlist from yesterday's close. The original reading is kept as stability configuration 11.
* **Peeks, disclosed:** the planner looked at NAUT's own trade and at raw signal counts on two batches; a smoke run on 2 of the 58 batches (393 stocks, 62 trades, mean +0.38R) was seen before the full run. No parameter was changed after either. The full run contains those same stocks.
* **Entry and exit:** signals are read at the close; entry is the next open; a stop that is gapped through fills at the open; the stop is checked first on a bar that could hit both; the trailing stop only ratchets up and uses the previous bar. A second, independently written implementation of this order model agreed with the module on **3,754 real-data trades** (3 exit models).
* **Controls (exact, not sampled):** for each trade, the average result of **every** liquid stock entered on that same date with the same exit (C3), narrowed to the same volatility bucket (C3v, the pre-registered comparison, because R-units make a flat 1% cost cheaper for volatile stocks), and of every stock in the bottom state that day without the trigger (C2).
* **Intervals:** bootstrap clustered by signal month (primary, because signals cluster in time), by half-year, and by stock. The verdict needs the month **and** half-year intervals to clear zero.

## Results

| | trades | stocks | mean R (net) | 95% interval (month) |
|---|---:|---:|---:|---:|
| **C0: bottom, then trigger (the rule)** | 1,568 | 911 | **+0.096** | -0.112 to +0.320 |
| C1: trigger alone, no bottom state | 68,370 | 4,287 | -0.175 | |
| Matched random entries, same date (C3) | | | -0.051 | |
| Matched random entries, same date and volatility (C3v) | | | +0.047 | |
| Bottom-state stocks on random days, no trigger (C2) | | | +0.088 | |

| C0 minus control | difference | interval (month) | interval (half-year) |
|---|---:|---:|---:|
| vs C3v (**pre-registered**) | +0.048 | -0.083 to +0.181 | -0.068 to +0.167 |
| vs C2 (does the trigger add anything?) | +0.009 | -0.130 to +0.149 | -0.120 to +0.134 |
| vs C3 (date only; not the right control, see above) | +0.146 | +0.019 to +0.276 | +0.024 to +0.265 |
| C1 vs C3v (trigger without the bottom) | -0.034 | -0.051 to -0.016 | |

Reading it: on its own a volume breakout is slightly **worse** than random entry. Inside a bottomed stock it is about the same as random entry. The one comparison that looks significant (vs C3) is the wrong control: it does not match volatility, so it mostly measures that cheap, volatile stocks pay less cost in R.

**Payoff shape of the rule:** win rate 32.2%, average win +2.17R, average loss -0.89R, median -0.71R, profit factor 1.16, 6.7% of trades at +3R or better. Exits: 875 trailing stop, 515 initial stop, 178 gapped through the stop (11%).

**Pre-registered verdict:** checks 1 (net expectancy above zero) and 3 (beats the bottom-only control) pass; check 2 (excess over the volatility-matched control, with the month and half-year intervals above zero) and check 4 (positive after the survivorship adjustment) fail. **No demonstrated edge.**

## Why the headline does not hold

| Diagnostic (after the fact; not part of the verdict) | mean R |
|---|---:|
| All 1,568 trades | +0.096 |
| Without the 5 biggest | +0.038 |
| Without the 10 biggest (0.6% of trades) | **-0.002** |
| Without the 20 biggest | -0.064 |
| 2020 only (n = 173) | +0.882 |
| Without 2020 | **-0.002** |
| Without 2009 and 2020, the two post-crash rebounds | -0.022 |

Years are erratic: 2020 +0.88, 2009 +0.22, 2023 +0.21, 2025 +0.29, but 2018 -0.67, 2019 -0.44, 2021 -0.25, 2026 -0.35. The three eras all show about +0.10 (2001-09 n=261, 2010-17 n=194, 2018-26 n=1,113), but that agreement is the same few rebound years each time. The biggest trades are real data, not errors (e.g. HMNY +21.7R is a reverse-split-adjusted 2017 spike; CAPR +17.2R and AGIG +20.4R are one-to-two-day spikes; AGIG was a short squeeze with a 20M-share day). The AGIG fill is optimistic: the model fills the trailing stop at 108.45 inside a bar that ranged from 166 to 59, which a real halted stock would not guarantee.

**Stability checks (never used to select):** all 8 variants have a positive excess over the matched control (+0.015 to +0.137R) and none has an interval above zero. Best: relative volume over 3, +0.137R (-0.033 to +0.308). Cost at 2%: mean +0.017R. Secondary exit (fixed 3R target, 60 bars): mean +0.115R, excess -0.013R. By entry price, exploratory only: $1-3 +0.22R (n=200), $3-10 +0.17R (n=498), $10 and up +0.03R (n=870).

## Survivorship

The cache holds only stocks still listed today. The stocks that bottomed and kept falling are exactly the ones missing, so every number above is an **upper bound**. Using the delisted list (common stocks only; 2,376 SPAC, warrant and unit rows and all ETFs removed), the share of stocks alive in a signal year that are now delisted is **14.1% (signal years 2013 onward) and 17.8% (all years)**. That is a *lower* bound for this rule, since stocks 70% down delist far more often than average.

If every missing signal lost 1.0R, the rule's mean falls to **-0.06R**; at 1.3R, to **-0.10R**. Break-even is a missing share of only **8.7% (at -1.0R) or 6.9% (at -1.3R)**. That assumption is deliberately harsh. As a gentler illustration, if missing signals behaved like an ordinary breakout (C1, -0.175R) the mean would be about **+0.06R**. So the survivorship-adjusted sign is unresolved, which is the case for getting delisted prices (see `docs/SETUP_LEARNING_2026-10-07.md`).

## NAUT, as a check on the rule

The rule fires on **2026-09-29**: previous close 74.3% below its 2-year high; 126-day low 0.841 against 0.62 before; breakout close 1.365 over the prior 20-day high of 1.17 on 18.6x relative volume. Entry 2026-09-30 at 1.46, stop 1.21, exited 2026-10-06 at 1.89 on the trailing stop: **+1.65R net**. The original same-bar reading would not have flagged it. On today's cache the rule flags nothing on 10-06, 10-05, 10-02 or 09-30; the only other hit in the last six sessions was ATOM on 10-01. The rule fires about 60 times a year across the whole market.

## What this does not test

* **The hand-picked turnaround story.** There are no historical fundamentals in our data (cash against market cap, burn rate, catalysts, insider buying), so the part of your idea that picks the stock cannot be tested here. This study says chart conditions alone do not provide the edge.
* **The 4-hour chart.** Only daily bars exist for 2000-2026, so the "daily/4h entry" is tested as a daily entry.
* **Portfolio effects.** Every trade risks one R; there is no cap on simultaneous positions.
* Yahoo split adjustment is retroactive (a stock that was $0.15 and later reverse-split shows as $3 in the past), so the "Price over $1" filter is imperfect. Dividends are not adjusted. The cost is one flat 1% assumption.

## The Finviz-style rule engine

`setup_engine/screen.py` is the reusable piece. 27 filters, 259 dropdown-style options, any number accepted (`Over $7.50`, `0-4% below`):

```python
from setup_engine.screen import Screen, indicators, clean_daily
Screen({"52-Week High": "70% or more below", "Relative Volume": "Over 2"})
Screen.from_text("RSI (14): Oversold (30); Price: Under $5; SMA200: Price above SMA200")
```

Every indicator uses only bars up to the signal bar (tested by truncating history), a missing value never matches, and an unknown filter or option raises an error that lists the valid choices. Screen today's market from the command line (reads the local cache; `--prior-rules` applies a rule to the previous bar):

```bash
python scripts/run_screen.py --rules "52-Week High: 70% or more below; Relative Volume: Over 2"
python scripts/run_screen.py --list-filters
```

The cache's last date is printed on every run. The 10-06 bar exists for only 8,288 of about 11,400 stocks (the rest are dropped as stale), so "latest" currently covers about three quarters of the market.

## Reproduce

```bash
python -m pytest tests/test_setup_engine tests/test_study -q     # 306 tests
python scripts/study_turnaround.py                               # ~80 s; appends 13 ledger entries
```
Needs `pyarrow` and, for speed, `numba` (both research-only; tests skip without them). A pure-Python fallback gives identical results.
