# Focusing on today's blue chips: what the data says

**Date:** 2026-10-12 · **Status:** research result, not trading advice · **Code:** `scripts/exploratory/bluechip_*.py` (exploratory, untested; every number below reproduces from them) and the tested daily screen `scripts/bluechip_daily.py`

## Decision and the one trap

The universe is the **current S&P 500** (503 names with sector, sub-industry, SEC identifier and join date, parsed from Wikipedia by `setup_engine/universe.py`). For blue chips the survivorship problem that blocked the earlier studies is much smaller, so delisted prices are not needed.

The trap is **hindsight**: today's members are today's winners. Average 6-month forward return per half-year, by era:

| Universe | 2001-09 | 2010-17 | 2018-26 |
|---|---:|---:|---:|
| Today's S&P 500 members, all history (hindsight) | +6.0% | +8.6% | +6.7% |
| Today's members, counted only from their join date | +4.0% | +7.3% | +5.4% |
| Top 500 by dollar volume **at the time** | +3.9% | +6.4% | +4.9% |
| All stocks | +4.2% | +5.4% | +1.5% |

Judging rules on today's list across all history flatters them by about 2 points per half-year (4-5% a year). Counting a member only from its join date removes most of it. The tests below use the top-500-at-the-time universe and the join-date-aware one.

## Price setups add little in blue chips

Excess return per half-year over the same-date blue-chip average, 95% interval (stock-clustered):

| Setup (top 500 at the time) | 2001-09 | 2010-17 | 2018-26 |
|---|---:|---:|---:|
| Stage-2 trend | +0.6% [-0.4, +1.6] | +0.7% [-0.0, +1.5] | **+1.8% [+0.9, +2.7]** |
| Tight base breakout | +0.3% [-0.3, +0.9] | +0.2% [-0.3, +0.7] | +0.7% [+0.1, +1.3] |
| Compression + golden cross | -0.3% [-1.0, +0.3] | +0.5% [-0.1, +1.0] | +0.7% [+0.0, +1.5] |
| RS leader (exploratory) | -0.8% [-1.8, +0.3] | +0.6% [-0.4, +1.5] | +2.1% [+1.0, +3.3] |
| Pullback in uptrend, 3 variants (exploratory) | -0.1% to +0.8% | -0.7% to +0.3% | +0.4% to +1.6% |
| Sector rotation, top-3 minus bottom-3 sectors | +0.1% [-1.3, +1.5] | -0.8% [-2.0, +0.5] | +1.6% [+0.0, +3.0] |

On all stocks the same trend setups gave +4.6% (2018-26); that premium came mostly from small and mid caps. About ten tests were run here, so one interval just above zero is not evidence. The RS-leader and pullback rows are exploratory (fixed before looking, but not pre-registered), and in "today's S&P 500" with hindsight the RS-leader premium looked like +3.3% against +1.4% when counted from join dates.

## Market regime is a risk control, not an entry signal

SPY weekly close above its 30-week average ("regime on") was **not** better for buying: after 2009, blue chips returned more over the next six months with the regime **off** (2010-17 +11.7% vs +5.6%; 2018-26 +9.7% vs +3.5%), because off periods followed drawdowns and were followed by rebounds. It protected only in 2001-09 (32% of six-month holds lost 10%+ with it off, vs 22% on).

## What to build instead

For blue chips the leverage is fundamentals (quality, valuation, earnings) and risk management (exits, sizing), not chart patterns. The daily screen (`scripts/bluechip_daily.py`) therefore reports chart state as context with a caveat, and adds a fundamentals snapshot. The SEC provider (`data_engine/sec_edgar.py`) is built to test the fundamentals half historically once a contact for the SEC's required header is set.

## Today's screen (as of 2026-10-09)

SPY is 5.6% above its 30-week average; 47% of 499 screened stocks are above theirs. One full-confluence name (HUM), 15 not-yet-extended leaders (MSFT, V, ABBV, ADP, EXPD, CPAY, TMO, PM, DGX, SOLV, TGT, FDS, WAT, BIIB, SNPS), 7 extended leaders (MRNA, P, HPE, CRWD, ILMN, DELL, MPC), no beaten-down reclaim. Leadership is in Information Technology (28 of 74), Health Care (21 of 60) and Energy (10 of 21). Outputs are in `~/.roboalgo/reports/bluechip_<date>.{csv,md,json}`.

## Caveats

* Today's S&P 500 sectors are the current labels. Ranks are within the S&P 500, an elite, survival-filtered set.
* The top-500-at-the-time universe still misses blue chips that vanished (Lehman, Wachovia and similar), so 2008 is flattered.
* The weekly screen changes only on Fridays.
