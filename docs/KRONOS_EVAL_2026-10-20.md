# Kronos on US stocks, 2026-10-20

A pre-registered, zero-shot test of the open-source Kronos candlestick foundation model
(github.com/shiyu-coder/Kronos) as a cross-sectional forecaster of 20-day US stock returns. Research
only: nothing here is wired into the app, and no strategy changed as a result.

## Verdict

**No demonstrated edge.** Mechanical rule output, verbatim: `no demonstrated edge`.

The pre-registered rule needed two things: the 95% lower bound of the Kronos mean rank IC above zero in
the clean window, and the 95% lower bound of (Kronos IC minus momentum IC) above zero. Neither held.
Kronos's mean IC over 14 clean origins was -0.017 with an interval from -0.136 to +0.105, and it lost to
12-1 momentum by 0.031 IC (interval -0.268 to +0.221). The window is not underpowered by the stated rule
(14 origins, threshold 12), but 14 monthly observations cannot resolve a small edge either. This is a
null result, and it is consistent with every other candle-based rule tested in this project.

## What was tested (pre-registration, fixed before any data was seen)

* **Model.** Kronos-small (`NeoQuasar/Kronos-small`, 99 MB) with Kronos-Tokenizer-base
  (`NeoQuasar/Kronos-Tokenizer-base`), zero-shot, no fine-tuning, no prompt tuning. Lookback 400 daily
  bars, horizon 20 trading days, T=1.0, top_p=0.9, sample_count=5. The forecast return is the mean over
  the sample paths of (predicted close at day 20 / last observed close) - 1. `predict_batch` averages
  the five decoded paths inside the model, so the mean ratio and mean(close at day 20)/last close - 1
  are the same number.
* **Origins.** Every month-end trading day. CLEAN window: origins from 2025-07-31 through the last
  month end with 20 trading days of realized forward data. Kronos's weights went public 2025-06-30 and
  its pretraining range is not disclosed, so US data before 2025-07-01 may have been seen in training
  and is treated as contaminated. INFORMATIONAL window: month ends 2023-01 to 2024-12, reported
  separately, never part of the verdict.
* **Universe at each origin.** Top 500 US stocks by 60-day average dollar volume computed from data up
  to and including the origin, last close >= $1, at least 400 bars of history. Only bars up to the
  origin close enter the filter or the model input.
* **Realized return.** Close at origin + 20 trading days (in that stock's own series) / close at the
  origin - 1.
* **Metrics per origin.** Spearman rank IC between forecast and realized return, top-decile minus
  bottom-decile mean realized return, and top-decile minus universe mean. Decile size is
  max(1, floor(n * 10% + 0.5)).
* **Baselines.** 12-1 momentum (close[t-21]/close[t-252] - 1), 1-month reversal
  (-(close[t]/close[t-21] - 1)), and a shuffled-rank placebo with a fixed seed.
* **Statistics.** Month-clustered bootstrap, 2000 draws, seed 0, 95%, reusing
  `setup_engine.trade_stats.cluster_bootstrap` and `cluster_bootstrap_diff`.
* **Verdict rule.** PASS only if both lower bounds are strictly above zero. A lower bound of exactly
  zero is not a PASS. UNDERPOWERED is reported when there are fewer than 12 clean origins.

## How it ran

* Kronos commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13), cloned to
  `~/.roboalgo/vendor/Kronos`.
* Separate interpreter `~/.roboalgo/kronos-venv` (Python 3.14.8) with torch 2.14.1, numpy 2.5.4,
  pandas 3.0.6, pyarrow 26.0.0, einops, safetensors, huggingface_hub, tqdm. The project venv was not
  touched.
* Weights loaded with `safetensors.torch.load_file` into a locally constructed module from `config.json`.
  No pickle load anywhere in the path.
* Device: MPS. Seed 20261020 set once before loading, so model sampling is reproducible.
* Cache ends 2026-10-06; 6,730 trading days; 38 usable origins (14 clean, 24 informational; 2 month
  ends were dropped as immature). 500 names at every origin, 19,000 forecasts over 796 distinct symbols.
* Timing: the 20-stock, 1-origin smoke test took 9 s end to end, of which 8 s was forecasting (0.40 s
  per name; 0.45 s per name including the model load). The full 19,000 forecasts took 9,208 s
  (2 h 33 min, 0.48 s per name). Metrics take about 5 s.
* Forecasts are cached per (symbol, origin) in `~/.roboalgo/kronos/forecasts.parquet`. A rerun with the
  cache filled finishes in 5 s, never imports torch, and reproduced every number below exactly.

## Clean window (2025-07-31 to 2026-08-31, 14 origins)

| signal | mean rank IC | 95% CI | origins |
|---|---|---|---|
| Kronos | -0.0171 | [-0.1364, +0.1053] | 14 |
| 12-1 momentum | +0.0142 | [-0.1218, +0.1456] | 14 |
| 1-month reversal | -0.0220 | [-0.1092, +0.0591] | 14 |
| shuffled placebo | -0.0039 | [-0.0258, +0.0174] | 14 |

Kronos IC minus momentum IC: **-0.0313 [-0.2676, +0.2214]**.

| signal | top decile minus bottom decile | top decile minus universe mean |
|---|---|---|
| Kronos | -1.63% [-9.68%, +7.51%] | -0.07% [-2.92%, +2.93%] |
| 12-1 momentum | +1.29% [-8.30%, +9.39%] | +1.27% [-5.19%, +7.10%] |
| 1-month reversal | -0.58% [-5.19%, +3.90%] | +0.00% [-2.40%, +2.13%] |
| shuffled placebo | -0.52% [-1.67%, +0.75%] | +0.06% [-0.56%, +0.84%] |

The placebo sits at about zero with a narrow interval, which says the pipeline itself is not
manufacturing an IC. The real signals are not distinguishable from zero, and neither is their
difference, because 14 monthly observations of a cross-sectional statistic that swings between -0.37
and +0.55 cannot pin a mean down.

Decile spreads are **gross**. No transaction costs, no slippage, no borrow. A monthly top-decile
strategy would pay all of them, and a 500-name monthly rebalance is not free.

## Informational window (2023-01 to 2024-12, 24 origins, possibly seen in training)

| signal | mean rank IC | 95% CI | origins |
|---|---|---|---|
| Kronos | -0.0262 | [-0.0768, +0.0220] | 24 |
| 12-1 momentum | +0.0628 | [+0.0030, +0.1251] | 24 |
| 1-month reversal | -0.0213 | [-0.0824, +0.0409] | 24 |
| shuffled placebo | -0.0014 | [-0.0208, +0.0194] | 24 |

Kronos IC minus momentum IC: -0.0889 [-0.1949, +0.0127].

Momentum is the only signal here whose interval excludes zero, and it does so barely. Kronos is again
slightly negative, and again behind momentum.

## Contamination diagnostic

* Kronos mean IC, 2023-2024 (possibly in training): -0.0262 [-0.0768, +0.0220].
* Kronos mean IC, clean window: -0.0171 [-0.1364, +0.1053].
* Old window minus clean window: -0.0091 [-0.1448, +0.1267].

The old window is not higher. There is no sign in these numbers that remembering the training data
helped the model predict US stocks, which is what a null result both ways looks like. Had the old
window been clearly stronger, that would have been consistent with training-data leakage and nothing
more could have been concluded from it. That is not what happened.

## What the model actually does (descriptive, not a metric of the test)

Two behaviours showed up consistently and are worth recording:

* **A standing pessimism.** The median forecast return was between -2.4% and -7.8% at every one of the
  14 clean origins, while the median realized 20-day return was between -9.1% and +3.1%. On these US
  daily bars, zero-shot Kronos predicts a decline for the typical large cap almost every month. A
  constant offset does not affect a rank IC or a decile spread, but it does mean the model's raw output
  cannot be read as an expected return.
* **The forecast is close to inverse momentum.** Across the clean origins, the mean rank correlation
  between the Kronos forecast and 12-1 momentum was -0.65 (range -0.40 to -0.78), and the per-origin IC
  series of the two signals had a correlation of -0.90. Kronos on these inputs is largely a
  mean-reversion ranking of the previous year's winners. It is not a new source of information about
  the cross-section, and that is the mechanical reason the "beats momentum" leg of the verdict was
  never going to clear.

One exploration note: the mean decile spread is driven by a few months. Dropping the single worst month
(2025-08-29, -22.5%) leaves a mean spread of -0.03%, and the largest positive month (2026-06-30, +46.5%)
is one observation with a +0.55 IC. Per-origin detail is in the table below.

## Per-origin detail

| window | origin | names | median forecast | median realized | IC Kronos | IC momentum | IC reversal | IC placebo | Kronos top-bottom |
|---|---|---|---|---|---|---|---|---|---|
| clean | 2025-07-31 | 500 | -3.96% | +2.43% | +0.130 | -0.138 | +0.136 | -0.032 | +3.34% |
| clean | 2025-08-29 | 500 | -6.77% | +0.77% | -0.371 | +0.339 | -0.015 | +0.073 | -22.52% |
| clean | 2025-09-30 | 500 | -6.34% | -0.13% | -0.208 | +0.060 | -0.162 | -0.044 | -10.96% |
| clean | 2025-10-31 | 500 | -6.06% | -0.45% | +0.211 | -0.177 | +0.157 | -0.021 | +13.45% |
| clean | 2025-11-28 | 500 | -5.06% | +0.64% | +0.063 | -0.152 | -0.118 | +0.008 | +3.32% |
| clean | 2025-12-31 | 500 | -5.13% | +3.13% | -0.203 | +0.220 | -0.007 | +0.005 | -17.07% |
| clean | 2026-01-30 | 500 | -4.77% | +2.06% | -0.276 | +0.080 | -0.374 | -0.083 | -19.30% |
| clean | 2026-02-27 | 500 | -6.76% | -9.09% | +0.008 | -0.056 | -0.055 | +0.052 | +4.21% |
| clean | 2026-03-31 | 500 | -2.36% | +2.96% | -0.068 | +0.276 | +0.003 | +0.049 | -6.16% |
| clean | 2026-04-30 | 500 | -5.76% | -0.84% | -0.083 | +0.160 | -0.199 | -0.010 | -7.39% |
| clean | 2026-05-29 | 500 | -7.80% | +1.20% | +0.059 | +0.024 | +0.147 | +0.013 | -2.80% |
| clean | 2026-06-30 | 500 | -4.32% | +0.35% | +0.554 | -0.597 | +0.181 | -0.013 | +46.52% |
| clean | 2026-07-31 | 500 | -4.48% | +1.59% | +0.158 | -0.276 | -0.141 | -0.058 | +4.01% |
| clean | 2026-08-31 | 500 | -5.78% | -4.40% | -0.213 | +0.436 | +0.137 | +0.007 | -11.51% |
| info | 2023-01-31 | 500 | -3.95% | -3.57% | -0.139 | +0.100 | -0.086 | -0.037 | -3.25% |
| info | 2023-02-28 | 500 | -3.82% | -2.78% | -0.005 | +0.174 | -0.039 | +0.062 | -4.57% |
| info | 2023-03-31 | 500 | -3.82% | -0.23% | +0.052 | +0.202 | -0.066 | -0.031 | -2.54% |
| info | 2023-04-28 | 500 | -3.72% | -3.45% | -0.012 | +0.102 | +0.253 | -0.081 | -5.13% |
| info | 2023-05-31 | 500 | -1.89% | +6.25% | -0.040 | -0.008 | -0.033 | +0.001 | -1.39% |
| info | 2023-06-30 | 500 | -5.78% | +3.32% | +0.087 | -0.204 | -0.042 | +0.057 | -1.81% |
| info | 2023-07-31 | 500 | -6.12% | -4.59% | +0.115 | +0.071 | +0.214 | +0.001 | +6.96% |
| info | 2023-08-31 | 500 | -5.45% | -5.17% | -0.026 | -0.079 | -0.151 | -0.052 | -3.67% |
| info | 2023-09-29 | 500 | -1.96% | -5.80% | +0.035 | +0.001 | -0.210 | -0.051 | -0.83% |
| info | 2023-10-31 | 500 | +1.50% | +8.81% | +0.032 | +0.192 | +0.265 | +0.101 | -2.60% |
| info | 2023-11-30 | 500 | -3.26% | +6.37% | +0.120 | -0.165 | -0.280 | -0.047 | +8.46% |
| info | 2023-12-29 | 500 | -7.85% | +0.60% | -0.251 | +0.303 | +0.188 | -0.010 | -4.18% |
| info | 2024-01-31 | 500 | -5.41% | +4.24% | -0.199 | +0.339 | -0.076 | -0.010 | -6.74% |
| info | 2024-02-29 | 500 | -9.43% | +3.77% | +0.066 | -0.141 | +0.027 | +0.035 | -2.45% |
| info | 2024-03-28 | 500 | -11.20% | -4.22% | +0.015 | -0.085 | -0.176 | +0.070 | +0.81% |
| info | 2024-04-30 | 500 | -6.02% | +0.55% | -0.199 | +0.216 | -0.028 | -0.069 | -8.28% |
| info | 2024-05-31 | 500 | -9.31% | -1.52% | +0.092 | +0.081 | +0.062 | +0.015 | +4.07% |
| info | 2024-06-28 | 500 | -4.60% | +2.34% | +0.212 | -0.176 | +0.061 | -0.045 | +6.33% |
| info | 2024-07-31 | 500 | -7.10% | +1.47% | -0.113 | +0.088 | -0.059 | -0.039 | -3.20% |
| info | 2024-08-30 | 500 | -7.87% | +1.61% | +0.032 | +0.008 | +0.006 | +0.050 | +0.70% |
| info | 2024-09-30 | 500 | -9.07% | +0.36% | -0.078 | +0.160 | -0.084 | +0.049 | -1.43% |
| info | 2024-10-31 | 500 | -6.26% | +6.88% | -0.242 | +0.194 | -0.382 | -0.021 | -9.76% |
| info | 2024-11-29 | 500 | -11.90% | -6.83% | +0.006 | -0.079 | +0.019 | +0.068 | -2.90% |
| info | 2024-12-31 | 500 | -4.20% | +4.34% | -0.190 | +0.210 | +0.107 | -0.051 | -5.37% |

## Caveats

* **Survivors only.** The price cache is today's listings, so delisted stocks are missing. Over a
  15-month clean window that bias is small but it flatters every mean and every IC slightly.
* **About 14 clean origins.** Every interval here rests on 14 monthly observations. The intervals are
  wide and the test is weak: a zero inside an interval is not proof of no edge either. The failure to
  find an edge is not the same as proof that none exists, especially with a statistic that swings from
  -0.37 to +0.55 across months.
* **No costs.** Decile spreads are gross. No transaction costs, slippage or borrow, and a monthly
  top-decile rebalance of 500 large caps pays a real amount of them.
* **Zero-shot only.** No fine-tuning, one horizon (20 days), one lookback (400 bars), one sampling
  setting. This says nothing about a fine-tuned Kronos or a different configuration.
* **US large caps only**, the part of the market where candle patterns are most arbitraged away, so a
  null here does not rule out an edge in smaller or less liquid names.
* **Sampling is stochastic.** Seeded for reproducibility, but a different seed moves the numbers a
  little.
* **Data quality.** The pre-registered filter has no jump or price-cap test, so one symbol (TXGE) sits
  in every universe with a last close above $1,800 (rising to $7,700 by 2026) and an implausible dollar
  volume above $100bn a day. The rule was not changed after seeing it. It is one name of 500 and does
  not drive the IC, but it is a reminder that the cache is not clean.
* **No dividend adjustment**, and Yahoo's split adjustment is retroactive.

## Verification

* `tests/test_study/test_kronos_eval.py`: 21 tests, all passing, with no torch, no network, no cache.
  They cover look-ahead (a future bar cannot move the universe or the model input), universe boundaries
  (399 bars out, 400 in, $0.99 out, $1.00 in, the 60-bar window, the top-N cut), the IC sign and a
  hand-computed case, ties, the decile spread, the verdict boundary (a lower bound of exactly 0 is not
  a PASS), the UNDERPOWERED flag, the placebo at about zero, strict JSON with atomic writes, and that
  the module imports with torch unavailable.
* The whole `tests/test_study` directory: 270 tests passing.
* Mutation check, 8 deliberate bugs inserted one at a time, each caught, file restored byte for byte:

| mutation | result | caught by |
|---|---|---|
| flip the IC sign | caught | `test_ic_sign_is_what_the_metric_means` |
| feed the origin+1 bar to the model | caught | `test_forecast_input_ends_at_the_origin_and_ignores_the_future` |
| verdict uses >= 0 instead of > 0 | caught | `test_verdict_boundary_zero_lower_bound_is_not_a_pass` |
| drop the $1 price filter | caught | `test_universe_bar_and_price_boundaries` |
| wrong realized horizon (21 bars) | caught | `test_realized_return_runs_to_origin_plus_twenty_and_skips_the_origin_itself` |
| swap the top and bottom decile | caught | `test_decile_spread_top_minus_bottom_and_universe` |
| realized return starts one bar before the origin | caught | `test_realized_return_runs_to_origin_plus_twenty_and_skips_the_origin_itself` |
| universe dollar volume from the last 60 bars of the file | caught | `test_universe_ignores_every_bar_after_the_origin` |

Two of the mutations initially slipped through and were caught only after the tests were tightened: the
no-look-ahead test was vacuous (its origin had fewer than 400 bars, so both frames were empty) and its
synthetic future bar had the wrong column names. Both are fixed.

## What was not verified

* Nothing here tests a fine-tuned model, another horizon, another sampling temperature, or another
  market.
* No transaction cost model was applied, so no net return exists.
* The pretraining data of Kronos is undisclosed, so the clean window is a date cut, not a proof of
  out-of-sample status.
* The 2026 price cache is what it is. Where the data is implausible, as with TXGE, that was recorded
  and left alone rather than filtered after the fact.

## Reproduce

```
~/.roboalgo/kronos-venv/bin/python scripts/kronos_eval.py --smoke 20   # 20 names, 1 origin, timing only
~/.roboalgo/kronos-venv/bin/python scripts/kronos_eval.py              # full run, ~2.5 h cold, 5 s warm
/Users/asmian/RoboAlgo/venv/bin/python -m pytest tests/test_study/test_kronos_eval.py -q
```

Outputs land in `~/.roboalgo/reports/kronos_eval_<date>.json` and `.md`; forecasts cache in
`~/.roboalgo/kronos/forecasts.parquet`; universe statistics cache in
`~/.roboalgo/kronos/universe_stats_<data_end>_<n>.parquet`.

## Files

* `setup_engine/kronos_eval.py`: the fixed rules as pure functions. No torch.
* `scripts/kronos_eval.py`: the CLI, lazy torch, forecast cache, progress and ETA, report writer.
* `tests/test_study/test_kronos_eval.py`: the offline tests.
* This document.
