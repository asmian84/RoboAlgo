# Strategy-evolution fitness: audit findings

Scope: `strategy_evolution_engine/fitness.py` (`evaluate_genome_simple`, `compute_fitness`).
Date: 2026-10-06. Status: finding 1 fixed, finding 2 documented only (operator decision).

## 1. `entry_confluence_min` is inert (NOT fixed, by design)

`evaluate_genome_simple` reads `entry_confluence_min` into a local (`entry_min`) and never
uses it. Every bar from index 50 to `len-20` is a candidate entry; the only things that
decide a trade are `stop_atr_mult`, `target_atr_mult` and `hold_days_max`. The price input
(`{date, close, atr}` per bar, built in `pipeline/stage9_strategy_evolution.py`) contains no
confluence score at all, so the function could not apply the filter even if it tried.

Consequence: the evolution engine does not evolve entry logic. Every evolved "strategy" is a
stop/target/hold grid search on arbitrary entries, i.e. a random-entry baseline. The
`entry_confluence_min` value stored with a "best" genome is whatever random number the
genome happened to carry; it did not contribute to that genome's fitness.

### Evidence

Same series (seed 0, 2000 bars, close = 100 + cumsum(N(0,1)), atr = 2% of close),
stop=2.0, target=4.0, hold=20, only `entry_confluence_min` varied:

```
entry_confluence_min= 0: {'sharpe_ratio': -4.3217, 'win_rate': 0.2205, 'profit_factor': 0.5657, 'max_drawdown': 1.68, 'trade_count': 127}  fitness=0.0
entry_confluence_min=50: {'sharpe_ratio': -4.3217, 'win_rate': 0.2205, 'profit_factor': 0.5657, 'max_drawdown': 1.68, 'trade_count': 127}  fitness=0.0
entry_confluence_min=99: {'sharpe_ratio': -4.3217, 'win_rate': 0.2205, 'profit_factor': 0.5657, 'max_drawdown': 1.68, 'trade_count': 127}  fitness=0.0
```

Broader sweep: 500 random (stop, target, hold, series-length, drift) combinations, each
evaluated at `entry_confluence_min` in {0, 30, 50, 75, 90, 99, 100000, -5}: the returned
metric dict was identical across all 8 values in 500/500 cases. (The 400-bar seed-0 case from
the original report likewise gives the same 13-trade result at 0, 50 and 99.)

### Same problem, other genome fields

`pattern_type`, `regime_filter`, `position_size_pct` and `max_positions` are also never read
by `fitness.py`. Only 3 of the 8 genome parameters influence fitness:
`stop_atr_mult`, `target_atr_mult` and `hold_days_max`.
`mutation.py`, `population.py`, `genome.py`, the `strategy_genomes` table and
`GET /api/strategies/top` all carry these fields as if they mattered.

### What this means for stored fitness / "best genome" selection

- Ranking between genomes is driven only by stop/target/hold. Two genomes that differ only in
  `entry_confluence_min` (or pattern/regime) receive identical fitness; the "winner" among
  them is decided by sort order / tie-breaking, not merit.
- Genetic search over the inert parameters is pure drift: mutation and crossover move them
  around with no selection pressure.
- Any stored `strategy_genomes` row whose `entry_confluence_min` / `pattern_type` /
  `regime_filter` is presented as an evolved entry rule overstates what was learned.
- Fitness computed here is an in-sample score on a random-entry baseline with fixed-ATR
  exits. It should not be read as evidence of edge. (Related: trades that hit neither stop
  nor target inside `hold_days_max` are dropped from the statistics rather than closed at
  market, and exits are filled at exactly the target/stop price, not the bar's close.)

### Options (operator decision; nothing has been changed)

A. Wire confluence into entry selection. Requires a per-bar confluence score in the price
   input (stage 9 would have to load it from the confluence tables alongside close/atr),
   then skip bars where `confluence < entry_confluence_min` in the entry loop. This changes
   which genomes win and what the engine recommends, so previously stored genomes must be
   treated as superseded. Same treatment would be needed for `pattern_type` and
   `regime_filter` if they are meant to be live.

B. Remove the dead parameters. Drop `entry_confluence_min` (and the other unread fields) from
   `StrategyGenomeParams`, `MUTATION_RANGES`, `generate_initial_population`, and stop
   persisting/serving them (`StrategyGenome` columns, `api/routers/strategies.py`,
   `frontend/src/types/index.ts`). Honest about what the engine actually does, smaller
   search space, but it is a schema change (columns can stay nullable for old rows).

Until one of these is chosen, the read site in `fitness.py` carries an "INERT" comment that
points here.

## 2. Degenerate Sharpe / profit factor (FIXED)

Exits are fixed ATR multiples, so each trade return is one of two values (about +8% at a
4-ATR target, -4% at a 2-ATR stop when atr = 2% of close). When every trade shares one
outcome, or there is a single trade, return variance is 0 and the old
`mean / max(std, 1e-9) * sqrt(252)` manufactured a meaningless 1e8-scale value.

- Before: seed-0 400-bar run, 13 trades, all losses: `sharpe_ratio = -634980314.6555`.
  An all-win run produced `sharpe_ratio = 1269960629.311` and `profit_factor = 2960000000.0`
  (zero losses -> denominator floored to 1e-9).
- After: `sharpe_ratio = 0.0` when std < 1e-9 (undefined, not infinite); `profit_factor`
  is `PROFIT_FACTOR_CAP` (10.0) when there are wins but no losses, 0.0 when there is neither
  profit nor loss. `win_rate`, `max_drawdown` have no near-zero division.
- Degenerate runs are common: 1346 of 3000 random (genome, series) trials hit the
  zero-variance branch.

Fitness is deliberately unchanged. `compute_fitness` used to clip the +/-1e8 Sharpe to +/-1,
so all-win runs got full Sharpe credit and all-loss runs full penalty. Reporting 0.0 alone
would have changed that (an all-loss genome would have outranked a mostly-losing one). So
`evaluate_genome_simple` adds a `sharpe_undefined_sign` key (sign of the mean return) only in
the undefined case, and `compute_fitness` uses it in place of the neutral 0.0. A 3000-trial
old-vs-new sweep showed 0 fitness differences; only the raw `sharpe_ratio` and
`profit_factor` fields changed, and only in degenerate cases.

Note: Sharpe is still annualised with sqrt(252) over per-trade returns (not per-day returns),
so non-degenerate values can be large (max 51.7 in the sweep). That is a scaling choice, not
a division-by-zero artifact, and was left alone.

## 3. Persisted fitness values

Fitness IS persisted. Flow:

1. `infrastructure/celery_app.py` stage 9 -> `pipeline/stage9_strategy_evolution.py::run`
   (scheduled weekly, Sundays 02:00 ET in `infrastructure/scheduler.py`).
2. `StrategyEvolver.evolve` (`evolver.py:53-54`) calls `evaluate_genome_simple` +
   `compute_fitness` per genome per generation.
3. Top 10 are inserted as new rows in table `strategy_genomes` (`database/models.py`
   `StrategyGenome`): `fitness`, `sharpe_ratio`, `win_rate`, `profit_factor`,
   `max_drawdown`, `trade_count`, plus the genome parameters and `genome_data` JSON.
   Rows are only ever added, never recomputed or deleted.
4. `GET /api/strategies/top` (`api/routers/strategies.py`) serves them ordered by `fitness`
   descending, optionally filtered by `is_active`. The frontend hook `useTopStrategies`
   exists (`frontend/src/api/hooks.ts`) but no component currently calls it.

Silent correctness problems:

- Rows written before the one-trade-per-symbol fix have fitness/trade_count computed on
  about one trade per symbol. They are not comparable with rows written after it, yet the
  API sorts old and new rows together by `fitness`. Old rows can rank above or below new
  ones for reasons unrelated to quality. The raw `sharpe_ratio` in old rows can also hold
  +/-1e8-scale garbage, and `profit_factor` ~1e7-1e9.
- Every stored row, old or new, was produced by the random-entry baseline in finding 1.
- Dedup in `evolve()` is by `genome_id` within a single run only; across runs each run adds
  a new set of rows with its own fitness scale, so the table mixes populations.

State checked on 2026-10-06: the local database (`DATABASE_URL` default,
`postgresql://asmian@localhost:5432/roboalgo`) has the `strategy_genomes` table with 0 rows.
The deployed (Render) database was not accessible from this audit; it may contain rows.
No migration or deletion was performed. Suggested handling once the operator decides on
finding 1: either exclude pre-fix rows by a `created_at` cutoff in the API query, or truncate the
table and let the next weekly run repopulate it.
