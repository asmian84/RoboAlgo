# Finding 50x stocks: how rare, and what they looked like at the start

**Date:** 2026-10-12 · **Status:** research result, not trading advice · **Code:** `scripts/multibagger_study.py` (tested, mutation-checked) · **Raw output:** `docs/multibagger_study_results.json`

**SURVIVORS ONLY.** The price cache holds stocks listed today, so every rate below is an upper bound. For the fallen stocks examined here, the ones that died are exactly the ones missing.

## How rare

Eligible start = price at least $1 and at least $300,000 a day, every December 2001-2015, 120 months ahead (24,044 starts, 2,728 stocks):

| Reached within 10 years (peak) | share of starts |
|---|---:|
| 5x | 17.3% |
| 10x | 4.6% |
| 20x | 0.96% |
| **50x** | **0.07%** (12 distinct stocks) |
| 100x | 0.01% |

The 12: TSLA (2011-12, 200x), ENPH, AMD (2015-12, 89x), SHOP, DXCM, APPS, BB, DPZ, STRL, SIRI, NFLX, CLF.

## What the winners looked like at the start

Median price $2.36 (all stocks $24.50); 69% below the 60-month high (all: 20%; 75% if a full three years of history is required); down 31% over the prior year and 45% over three years (all: +9% and +28%); same dollar volume as everyone else; about twice the volatility. Precision of simple profiles at the start (share that went 50x): price under $5 1.22% (16x the base rate); under $5 and 70%+ below the high 1.57% (21x); up 100%+ in 12 months **0 of 1,408**.

## The fallen-stock pool (monthly starts 2001-2015)

Profile A = price $1-$10, 70%+ below the 60-month high, at least $300k a day:

| Variant | starts | stocks | 10x | 20x | 50x | fell another 50% in 3y | ended 10y below start | median end multiple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: profile only | 11,779 | 942 | 18.4% | 6.2% | 0.84% | 31.8% | 27.5% | 2.54x |
| B: A + monthly close above its 10-month average | 3,490 | 516 | 14.7% | 5.0% | 0.52% | 31.4% | 29.3% | 2.30x |
| C: B + up 20% in 3 months | 1,868 | 461 | 16.2% | 5.4% | 0.43% | 33.4% | 31.5% | 2.24x |
| Control: A still below its 10-month average | 8,289 | 875 | 19.9% | 6.7% | 0.98% | 32.0% | 26.7% | 2.66x |

**Waiting for a trend confirmation lowered the odds** (50x 0.52% against 0.98% for the still-below-trend control): the multiple is made early. The profile alone is a long shot (0.84%), and 31% of it fell another 50%. These survivor-only numbers are optimistic: a stock that fell 70%, sits under $10 and later went bankrupt is not in the data.

## Today's pool

521 stocks match the profile at the last complete month (2026-09). **36 carry a suspected unadjusted split or spin-off in the last three years** (WHLR and FFAI among them), and many show a drawdown of essentially -100% (JAGX, EDBL, DBGI, CDT, DCX, GRI): serial reverse-splitters whose adjusted history makes them look fallen. A crude survival-and-turnaround checklist (size, growth, margins, cash or positive free cash flow, above trend) passed only 8 of an earlier 328-stock version and let through a crypto-holding company and a heavily indebted legacy firm. **No edge has been shown for this pool.**

## What it means

* 50x is about one stock in 1,400 starts, and the winners began fallen and cheap. The profile raises the odds 20-fold and they are still about 1%.
* What separated AMD or TSLA from the 98% was a change in the business, not visible in the price. AMD in 2015 would have failed a "safe" screen (revenue down 26%, heavy debt).
* A real gem finder tests a survival-and-turnaround screen on starts that include the failures. That needs delisted prices (the Tiingo loader, `scripts/tiingo_load_delisted.py`, is ready) and fundamentals (`data_engine/sec_edgar.py`, needs a contact for the SEC header). Until then the honest framing is a basket of many small bets with wide exits.

## Reproduce

```bash
python scripts/multibagger_study.py --workers 8     # ~1 min; writes docs/multibagger_study_results.json and ~/.roboalgo/multibagger/pool_<month>.csv
```
After the delisted history is merged (`scripts/ingest_delisted_history.py` writes `terminal.json`), point `--history` at the merged folder: the output is then labelled survivorship-corrected and dead stocks are held at their last price.
