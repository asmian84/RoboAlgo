# Adversarial Validation Prompt — RoboAlgo overnight audit, 2026-10-06

Paste everything below the line into DeepSeek (or any other model) **with the repository
attached or checked out**. It is written to be adversarial on purpose: it asks the reviewer
to *disprove* claims rather than confirm them, and every claim carries the command that
tests it.

A note on why it is shaped this way. Over this project, several things looked correct and
were not: a breakout engine that had produced zero rows since inception because `int(NaN)`
was swallowed as a warning; a ticker universe that collapsed from 13,786 to 361 because
pandas parsed the ticker `NA` as `NaN`; a screener that ranked a stock delisted seven months
earlier as its top trade. In each case a plausible-looking summary was wrong and only a
command proved it. So: **do not accept any claim below because it is written down. Run the
command.**

---

# ROLE

You are an adversarial code reviewer auditing an overnight automated work session on a
**live trading system**. Your job is to find what is wrong, overstated, or unverified — not
to confirm good work. A wrong number in this codebase can cause a real losing trade, so
correctness outranks completeness, style, and feature count.

Assume the previous agent was competent but occasionally overconfident. It has been wrong
before in exactly one recurring way: **believing a measurement without checking whether the
measurement itself was sound.**

# SYSTEM UNDER REVIEW

RoboAlgo — algorithmic trading platform.
- Backend: Python 3.11, FastAPI, ~70,000 LOC, 170 `/api` routes across 53 routers, 72 `*_engine/` packages
- Frontend: React 19 + Vite 6 + Tailwind, live routes are only `/finder`, `/chart`, `/paper`, `/settings`, `/login`
- Data: PostgreSQL via SQLAlchemy 2.0; Celery + Redis for scheduling; deployed on Render
- Python interpreter: `/Users/asmian/RoboAlgo/venv/bin/python` (the venv lives in the MAIN
  checkout, not in the worktree — a bare `python` will not have the dependencies)

## The single most important invariant

The signal chain is strictly ordered:

```
prices → indicators → volatility → compression → breakout
```

Every engine **left-joins the one above it and skips bars whose upstream columns are NaN**.
A gap anywhere silently truncates everything downstream, at the date it stopped, **with no
error raised anywhere**. This system shipped for months with two stages unimplemented and
the only symptom was a date that quietly stopped moving.

**Therefore:** any change you review that touches ingestion, indicators, or scanning must be
judged on whether it can introduce a silent gap. "It didn't crash" is not evidence of
correctness here. Ask specifically: *if this failed, how would anyone find out?*

---

# PART 1 — CLAIMS TO FALSIFY

For each claim: run the command, record actual output, mark
**CONFIRMED / REFUTED / UNVERIFIABLE**, and for anything not CONFIRMED say what the truth is.

### CLAIM 1 (CRITICAL) — TradingView's precomputed indicators are corporate-action-unsafe

Asserted: TradingView's screener returns indicator columns that are not adjusted
consistently with the `close` column, making them unusable for signal math. Evidence given:
`NYSE:CTVA` returns `close=12.39, SMA50=77.32, SMA200=77.88, RSI=6.38, change=+3.94%` — i.e.
a stock reported as 84% below its own 50-day mean at RSI 6 (screaming oversold) while
closing **up** 3.94% on the day. `SPY`, `MSFT`, `NVDA`, `AAPL` are self-consistent, so the
corruption is **silent and symbol-dependent**.

```bash
/Users/asmian/RoboAlgo/venv/bin/python - <<'EOF'
import requests
cols=["name","close","SMA50","SMA200","RSI","change"]
r=requests.post("https://scanner.tradingview.com/america/scan",
  json={"symbols":{"tickers":["NYSE:CTVA","AMEX:SPY","NASDAQ:MSFT","NASDAQ:AAPL"],"query":{"types":[]}},"columns":cols},
  headers={"User-Agent":"Mozilla/5.0"},timeout=20)
for row in r.json()["data"]:
    print(row["s"], dict(zip(cols,row["d"])))
EOF
```

**Then audit the consequence, which matters more than the claim:**
`data_engine/tradingview_provider.py` must request **only** `name`, `close`, `volume`.
```bash
grep -n "columns" data_engine/tradingview_provider.py
grep -rn "SMA50\|SMA200\|\"RSI\"\|'RSI'" data_engine/ scanner_engine/
```
If any TradingView indicator column reaches `scanner_engine/finder.py`'s signal logic, that
is a **CRITICAL wrong-trade defect** — report it as the top finding.

Also check the sampling: is CTVA a one-off, or are there many such symbols? Query a few
hundred tickers and count how many have `close/SMA50` outside [0.5, 2.0]. A higher rate makes
this more serious, not less. Report the actual rate.

### CLAIM 2 (CRITICAL) — compression and breakout were scheduled by nothing

Asserted: `range_compressions` and `breakout_signals` — the two tables the Finder reads —
were in no schedule at all. `stage_map` in `infrastructure/celery_app.py` covers stages
1–11 and neither table is a numbered stage, so even a perfectly healthy Celery deployment
would have left the screener frozen forever. A `run_range_engines` task was added.

```bash
grep -n "stage_map" -A 15 infrastructure/celery_app.py
grep -n "run_range_engines" infrastructure/celery_app.py infrastructure/scheduler.py
git log -p --follow infrastructure/scheduler.py | head -200
```
Verify by reading the **pre-change** version in git history that neither engine was
scheduled. Then confirm the new task runs compression **before** breakout in a single task
(breakout left-joins compression; a 5-minute gap between two separate beat entries is a
guess, not a guarantee).

### CLAIM 3 (HIGH) — the schedule fired before the market close

Asserted: beat crontabs were UTC while commented "after US market close". 18:30 UTC is
14:30 ET in summer and 13:30 ET in winter; the close is 16:00 ET. So every "daily" bar
written was a **partial session**. Fixed by setting Celery to `America/New_York`.

```bash
grep -n "timezone\|enable_utc" infrastructure/celery_app.py
/Users/asmian/RoboAlgo/venv/bin/python -c "
import sys; sys.path.insert(0,'.')
import infrastructure.scheduler
from infrastructure.celery_app import app
from datetime import datetime; from zoneinfo import ZoneInfo
now=datetime.now(ZoneInfo('America/New_York'))
for n,e in app.conf.beat_schedule.items():
    print('%-28s %s' % (n,(now+e['schedule'].remaining_estimate(now)).strftime('%a %H:%M')))
"
```
Every daily ingestion entry must fire **after 16:00 ET**. Flag any that does not.
Then ask the harder question the previous agent should have asked: **is a partial-session
bar still sitting in the database from before the fix?** A fix that doesn't repair the
corrupted history it caused is half a fix. Check whether any historical bar was written
mid-session and whether anything backfilled over it.

### CLAIM 4 (HIGH) — a near-miss that would have deleted 3,057 lines of live code

Asserted: an earlier reachability analysis in this session used a regex requiring whitespace
after `import`, so it missed `lazy(() => import('...'))`. It therefore marked
`frontend/src/components/chart/KLineChartOverlay.tsx` (2,747 LOC) and
`components/chart/patternOverlays.ts` (310 LOC) as dead. **They are live**, reached from
`frontend/src/pages/FinderPage.tsx:12` on the `/finder` route.

```bash
grep -n "lazy\|import(" frontend/src/pages/FinderPage.tsx | head
grep -rn "KLineChartOverlay" frontend/src/
```
**This is the claim most worth attacking.** Re-run reachability yourself with a resolver
that follows BOTH `from '...'` and `import('...')`, and confirm the 25 deleted modules were
genuinely unreachable. If even one deleted file was reachable, that is a **CRITICAL
regression** — name it and give the restore command (`git checkout <path>`; restore point
tag is `pre-consolidation-2026-09-29`).

Also check for reachability edges that *both* analyses may have missed: route-level
`React.lazy`, `require()`, dynamic string-built import paths, re-exports via `index.ts`
barrels, and any component referenced only from a string map or config object.

### CLAIM 5 (HIGH) — fitness sampled at most one trade per symbol

Asserted: `strategy_evolution_engine/fitness.py::evaluate_genome_simple` had an
unconditional `break` in the outer loop, so every symbol contributed at most one trade.
Sharpe, win rate, profit factor and trade count were therefore computed on ~1 observation
per symbol, and `compute_fitness`'s `trade_frequency` term was driven by that artifact.

```bash
git diff strategy_evolution_engine/fitness.py
/Users/asmian/RoboAlgo/venv/bin/python -c "
from strategy_evolution_engine.fitness import evaluate_genome_simple
import numpy as np; rng=np.random.default_rng(0)
bars=[{'close':float(c),'atr':float(c)*0.02} for c in 100+np.cumsum(rng.normal(0,1,400))]
print(evaluate_genome_simple({'entry_confluence_min':60,'stop_atr_mult':2.0,'target_atr_mult':4.0,'hold_days_max':20},{'TEST':bars}))
"
```
Confirm `trade_count > 1` now, and — more importantly — that **trades do not overlap**
(each entry index must be after the previous exit). An "improvement" that double-counts
overlapping positions would inflate every metric and is worse than the original bug.

### CLAIM 6 (HIGH) — the validation gate rejects best-of-N noise

Asserted: `strategy_validation/gates.py` implements the Bailey & López de Prado deflated
Sharpe ratio, and a Monte Carlo test proves that the best of 500 pure-noise strategies is
**rejected**.

```bash
/Users/asmian/RoboAlgo/venv/bin/python -m pytest tests/test_strategy_validation -q
```
**Audit the math, do not trust the test.** A widely-circulated version of the deflated
Sharpe formula is wrong (it passes Sharpe 3.0 and rejects 1.8). Verify against the original
paper:
- `expected_max_sharpe` = `sqrt(V) * ((1-γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))`, γ ≈ 0.5772156649
- SR standard error = `sqrt((1 − skew·SR + ((kurtosis−1)/4)·SR²)/(n_obs − 1))`

Check monotonicity: more trials must *lower* DSR; higher observed Sharpe must *raise* it.
Then try to break it — feed it `n_trials=0`, `n_trials=1`, negative Sharpe, `n_obs=1`,
kurtosis below 1 (can make the variance term negative → `sqrt` of a negative), and NaN.
**Report any input that produces a NaN, a crash, or a spurious approval.**
Confirm it is NOT wired into the live trading path (that was deliberate — it would change
which strategies may trade real money, and that is the operator's decision):
```bash
grep -rn "strategy_validation" --include=*.py . | grep -v tests/ | grep -v strategy_validation/
```

### CLAIM 7 (MEDIUM) — duplicate 5-second polling was collapsed

Asserted: `MacroTickerBar.tsx` and `SideNav.tsx` each ran their own
`setInterval(fetchMacro, 5000)` against a yfinance-backed endpoint, causing
`YFRateLimitError` spam. Now one shared 15s poll via a context provider.

```bash
grep -rn "setInterval" frontend/src/components/layout/ frontend/src/context/
```
Start the API and frontend, load `/finder`, watch the server log for 60s, and **count**
`GET /api/macro` requests. Expect ~4 in 60s, not ~24. Report the actual count.

### CLAIM 8 (MEDIUM) — the screener no longer recommends delisted tickers

Asserted: `scan()` evaluated each instrument's latest stored rows with no recency check, so
`CFLT` — last traded 2026-03-05 — was being returned as the **top result** with a full
entry/stop/target built on a seven-month-old price. A `max_bar_age_days=5` gate was added.

```bash
/Users/asmian/RoboAlgo/venv/bin/python -c "
import sys,logging; sys.path.insert(0,'.'); logging.basicConfig(level=logging.WARNING)
from scanner_engine.finder import scan, FinderCriteria
from dataclasses import replace
r=scan(FinderCriteria(),limit=100); print('CFLT present:', 'CFLT' in [x['symbol'] for x in r])
r2=scan(replace(FinderCriteria(),max_bar_age_days=0),limit=100); print('gate off ->', 'CFLT' in [x['symbol'] for x in r2])
"
```
Then attack the threshold: **is 5 calendar days right?** Consider a Friday bar read on the
Tuesday after a Monday holiday (age 4), and a double-holiday week. Does the gate use
calendar days or trading days? Would a genuinely halted-but-not-delisted stock be wrongly
excluded, and is that the safe direction to fail? Check `scan_wide` too — does the
live-download path have the same protection, or only `scan()`?

### CLAIM 9 (CRITICAL) — the evolution engine does not evolve entries

Asserted: in `strategy_evolution_engine/fitness.py`, `entry_confluence_min` is read from the
genome and never used, so every bar is a valid entry. `pattern_type`, `regime_filter`,
`position_size_pct` and `max_positions` are **also** inert. Only `stop_atr_mult`,
`target_atr_mult` and `hold_days_max` affect fitness — so every "evolved strategy" is a
3-parameter grid search on **random entries**.

Claimed evidence: 500 random genome/series pairs × 8 values of `entry_confluence_min`
(0, 30, 50, 75, 90, 99, 100000, −5) produced **identical metric dicts in 500/500**.

```bash
grep -n "entry_confluence_min\|entry_min\|pattern_type\|regime_filter\|position_size_pct\|max_positions" strategy_evolution_engine/fitness.py
cat docs/fitness_findings.md
```
Reproduce the 0/50/99 comparison yourself. **If confirmed, judge whether the right call was
to document rather than fix** — fixing it changes which strategies the system recommends.
Then check the blast radius: `strategy_genomes` rows (`database/models.py:1203`) are written
weekly by stage 9 and served by `GET /api/strategies/top`, sorted by fitness across rows
computed under *different definitions of fitness*. Is that a correctness problem in a UI
anyone reads? The local DB has 0 rows; **the Render production DB was not checked.**

### CLAIM 10 (MEDIUM) — a live page calls an endpoint that does not exist

Asserted: `frontend/src/components/chart/KLineChartOverlay.tsx:321` requests
`GET /api/patterns/{symbol}/elliott-waves`, which matches no route in the 174-entry table.

```bash
grep -n "elliott" frontend/src/components/chart/KLineChartOverlay.tsx
grep -rn "elliott" api/routers/
```
Confirm, then **curl it** and report how the UI degrades — silently blank, console error, or
broken render? The audit did not test that.

### CLAIM 11 (MEDIUM) — the Settings page reports an unverified provider tier

Asserted: `/settings` displays AlphaVantage as "PREMIUM · 75 req/min", but the key is free
tier — AV rejects `TIME_SERIES_DAILY_ADJUSTED` with "This is a premium endpoint". The badge
is inferred from the `ALPHAVANTAGE_RPM` number the operator typed, not from the account.

```bash
grep -rn "PREMIUM\|req/min\|rpm" frontend/src/pages/SettingsPage.tsx | head
```
There is a "Test Connection" button. **Find out what it actually tests** — if it reports
success against a key that cannot serve daily bars, that is a worse defect than the badge.

### CLAIM 12 (MEDIUM) — the universe source is dead, and the TradingView path changes membership

Asserted: AlphaVantage `LISTING_STATUS` returns literally `{}` (2 bytes, HTTP 200), so the
yfinance fallback screens only **361** static names, not ~13,786. And TradingView's
`type == stock` **excludes all 6,703 ETF (`fund`) rows** while **including OTC**, which the
old AlphaVantage exchange-only listing never did — 340 of 861 penny survivors are OTC.

```bash
/Users/asmian/RoboAlgo/venv/bin/python -c "
import os,requests; from dotenv import load_dotenv; load_dotenv('/Users/asmian/RoboAlgo/.env')
r=requests.get('https://www.alphavantage.co/query',params={'function':'LISTING_STATUS','apikey':os.getenv('ALPHAVANTAGE_API_KEY')},timeout=30)
print(r.status_code, len(r.text), repr(r.text[:120]))"
```
**This is the judgment call worth your attention:** the scanner's universe silently changed
composition (ETFs out, OTC in). The audit says the UI's ETF preset is unaffected because it
uses the DB-backed `/scan`. **Verify that.** If any live surface now screens a different
asset population than its label implies, that is a user-facing correctness problem.

### CLAIM 13 (HIGH) — the delisted-ticker bug already produced a trade

Asserted: the paper account's only closed trade is **CFLT**, entered $30.60 / exited $30.78
on 2026-09-29 — a ticker whose last real bar is 2026-03-05.

```bash
curl -s localhost:8000/api/paper-account/trades?limit=10
```
Confirm. Then ask the question the audit did not: **can the auto-trader still open a position
in a stale symbol by another route?** The recency gate was added to `scan()` — check whether
the auto-trader's own symbol selection path goes through it, or around it.

### CLAIM 14 (CRITICAL) — a fresh deploy would have had no working database driver

Asserted: a bare `postgresql://` URL leaves the driver to SQLAlchemy; **2.0.x picks psycopg2,
2.1.x picks psycopg (v3)**. `requirements.txt` is unpinned (0 of 20 lines) and ships only
`psycopg2-binary`, so Render resolves SQLAlchemy 2.1.3 and every connection fails with
`ModuleNotFoundError: No module named 'psycopg'`. The suite never noticed, because
`import api.main` opens no connection and callers log the failure as a warning.

```bash
uv venv --python 3.11 /tmp/v311 && uv pip install --python /tmp/v311/bin/python -r requirements.txt
/tmp/v311/bin/python -c "
from sqlalchemy.engine import make_url; import sqlalchemy
print(sqlalchemy.__version__, make_url('postgresql://u@h/d').get_driver_name())"
git diff config/settings.py && /tmp/v311/bin/python -m pytest tests/test_config -q
```
Attack the fix: does rewriting the scheme at `config/settings.py` cover **every** place a
connection is made? (The audit found one engine, `database/connection.py`; look for others,
including scripts, Celery tasks and anything reading `os.environ["DATABASE_URL"]` directly.)
Is `postgresql+psycopg2://` still correct for the deployed SQLAlchemy? Then explain why a
suite that passed 267/9 on the production-equivalent environment still missed this, and
name what else in this repo fails the same way: **silently, at runtime, behind a
`try/except` that logs a warning**.

### CLAIM 15 (HIGH) — the committed `render.yaml` was invalid and would not have worked

Asserted: the committed file had 6 schema errors against Render's published Blueprint schema,
declared Postgres as a `pserv` (a private service, not a database), omitted the Supabase
variables the frontend needs at import time, and could not reach the API from the static
host because ~20 call sites hardcode relative `/api/...`.

```bash
curl -sSL -o /tmp/render.json https://render.com/schema/render.yaml.json
git show 9481965:render.yaml > /tmp/old.yaml   # the committed version that was validated: 6 errors
uv run --with jsonschema --with pyyaml python -c "
import json,yaml; from jsonschema import Draft202012Validator as V
s=json.load(open('/tmp/render.json'))
for f in ('/tmp/old.yaml','render.yaml'):
    print(f, len(list(V(s).iter_errors(yaml.safe_load(open(f))))), 'errors')"
```
Confirm 6 before and 0 after, then judge the **semantics** the schema cannot: is a
same-origin `/api/*` proxy rewrite the right design versus changing the call sites? Is
`ipAllowList: []` safe for this deployment? Is the `databases:` block safe to sync if a
database of that name already exists?

### CLAIM 16 (HIGH) — the Finder, replayed over the whole universe, has no edge

Asserted: replaying `scanner_engine/finder.py`'s rule over 10,631 listed tickers (20.0M ticker-days) and the
234 tracked instruments, no preset clears the gate. Best OOS portfolio Sharpe 0.42 vs SPY 1.29; the
deflated-Sharpe test needs ~2.0 after 13 trials. Full write-up: `docs/FINDER_VALIDATION_2026-10-06.md`.

```bash
python -m pytest tests/test_finder_backtest -q          # 28 tests, incl. parity with the production engines
python scripts/validate_finder.py && python scripts/gate_finder.py && python scripts/finder_sensitivity.py
```
**Attack the replay, because a backtest that is wrong in the optimistic direction looks like a good strategy,
and one wrong in the pessimistic direction looks like a bad one.** Specifically:
* Re-derive the Finder's rule from `scanner_engine/finder.py::_evaluate` and compare it with
  `strategy_validation/finder_backtest.py::build_signals`. Any condition missing or reordered?
* Look-ahead: is there ANY value at bar t that depends on bar t+1? (The tests mutate two such leaks and catch both. Find a third.)
* Is the order model fair? Stop-first on both-hit bars, gap-up fills at the open, no fill beyond the target,
  one order per setup. Is "one order per setup" too harsh or too kind? Does the 30-bar time stop bias results?
* The ledger holds 13 trials. Is that honest? Were more variants effectively tried while building this?
* The cost assumptions (0.10% / 0.40% / 1.00% round trip) are unmeasured. How sensitive is the verdict?
* Survivorship and reverse-split distortion are acknowledged. Is there a bias the report does NOT mention?
* The report originally blamed gap-through-stops, then was corrected after measurement: 55-57% of fills occur
  ABOVE the displayed entry, realised R:R ~0.68 mean. Verify that number independently from raw prices.

### CLAIM 17 (HIGH) — "learn the AMD/NVDA runner setup" found nothing that survives, in two rounds

Asserted: a learner searched daily/weekly/monthly features for setups that precede a 5x in 3 years (round one) and a
doubling in 6 months (round two). Round one's 17 learned rules were all rejected; round two validated **0 of 51**, and
**0 on three placebo seeds** (outcomes circularly shifted in time per stock). The learned rules are crash/capitulation
patterns whose flagged stocks mostly collapse. Full write-up: `docs/SETUP_LEARNING_2026-10-07.md`;
registry: `setup_engine/registry/registry.json`; append-only ledger: `setup_engine/registry/trials.jsonl`.

```bash
python -m pytest tests/test_setup_engine -q                 # 73 tests, incl. truncation-invariance and purge checks
python scripts/learn_setups.py --objective runner6m         # ~55 s, includes the 3 placebo runs
```
**A pipeline that says "no" can be broken in the pessimistic direction, so try to make it find something that is there.**
* Plant a real effect in synthetic data (a feature that truly predicts a run, strength you choose) and check the
  pipeline FINDS it. Does `learn.fit_rules` -> `evaluate` -> `robustness` have power, or does it reject everything?
* The placebo shows ~0 false positives. Is the placebo itself valid (circular shift per stock keeps each stock's marginal
  distribution but breaks timing)? Could it be too gentle or too harsh?
* Look-ahead: every feature must be computable from bars up to the observation date. The tests truncate the history
  and demand identical features. Find a feature that still uses a future bar (weekly/monthly bars that are incomplete at
  the observation date are the usual culprit).
* Splits are purged by label horizon (`learn.check_purged`). Is a gap missing anywhere, e.g. the relative-strength rank,
  which is cross-sectional and computed on the observation date?
* The ledger counts 371 trials and Bonferroni uses the per-round candidate count. Is that honest, given how many
  feature sets, depths and learners were tried while building? Is the 3-year objective's 5x threshold itself a tuned choice?
* "Timing lift" divides by the SAME stock's own base rate so a rule cannot win just by picking volatile stocks. Is the
  stock-level bootstrap (ratio of sums) correct when stocks contribute very different numbers of observations?
* Survivorship: acknowledged, and round two uses a 6-month horizon to reduce it. Is 6 months actually short enough?
  What does the delisted list (`~/.roboalgo/delisted/av_delisted.csv`, 7,539 US stocks) say about how large the hole is by year?

### CLAIM 18 (HIGH) — three trend setups out-earn the average liquid stock by +1% to +5% per half-year

Asserted: Stage-2 trend, tight base breakout and compression + golden cross (hand-written, fixed before any learning)
have timing lift BELOW 1 for doubling, yet their 6-month winsorised return exceeds the same dates' average in every
era (2001-12, 2013-17, 2018-26) and among stocks trading >= $10M/day. They are registered `approved`, `provisional`.
This is the only positive finding of the whole effort, so it deserves the hardest attack.
```bash
python scripts/register_setups.py --objective runner6m      # prints excess by era and liquid-only
```
* "Excess" = winsorised (1/99) 6-month return minus the same-date mean over all stocks in the panel. Recompute it
  independently from `~/.roboalgo/setups/obs.parquet`. Does the answer change without winsorising? With the median?
* The benchmark is an equal-weighted average of liquid LISTED stocks. Equal-weighting small stocks is a known source of
  spurious "premia". Does the effect survive against a cap-weighted or sector-neutral benchmark?
* The three rules select overlapping stocks. Is that one finding counted three times? How correlated are the flags?
* **Forking paths:** the textbook rules' excess return was first examined AFTER the learned rules failed. Is
  "provisional" a sufficient label, or should they be rejected until tested on data not yet seen?
* Costs: +1% to +5% per half-year is before costs, and the sub-$5 / thin names are where costs are largest.
  Quantify turnover and a realistic spread for a portfolio that holds these flags.
* Is the effect concentrated in a few years (the 2018-26 era is far stronger than 2001-12)? Is it just a bull-market beta tilt?

### CLAIM 19 (MEDIUM) — scans carry a `setup_evidence` tag that matches the learned definition exactly

Asserted: `scanner_engine/finder.py::attach_setup_evidence` (via `setup_engine/apply.py`) adds `setup_evidence` to every scan
result. It tags only: it never re-ranks, filters or drops. A scanned stock evaluated from 2 years of bars plus an
interpolated rank matches the full-history, exact-rank definition on **364 of 364** sampled stocks. Failures are visible:
`setup_evidence: null` plus `setup_evidence_error`, never an empty list that reads as "matches nothing".
```bash
python -m pytest tests/test_setup_engine/test_apply.py -q    # 13 tests; 7 mutants were run and all caught
```
* Find an input where `setup_evidence` is an empty list but the stock actually matches (a false negative that looks clean).
* Does `scan()` on stored bars and `scan_wide()` on freshly downloaded bars give the same tag for the same stock on the same day?
* `rs_reference.json` is a snapshot of 4,319 stocks, stale after 14 days. What does a stale or missing file do to the tag?
* Does the tag's wording ("earns a few percent more than average") stay honest when shown next to a stock the Finder
  has labelled "entry"/"target"? Is there a place where it could be read as a buy signal?

### CLAIM 20 (MEDIUM) — delisted prices are not obtainable free, and the ingest path is bias-safe

Asserted: Yahoo returned history for 4 of 40 known delisted tickers (all re-used symbols); AlphaVantage, FMP and Finnhub
gate delisted prices; Stooq and Nasdaq Data Link refuse anonymous access. `scripts/ingest_delisted_history.py` skips
re-used tickers, marks every ingested stock terminal, and never modifies the survivor cache; the dataset builder holds
a terminal stock's last price flat (`features.hold_at_last_price`) so its collapse stays in the data.
```bash
python -m pytest tests/test_ingest -q          # 5 tests (need pyarrow; they skip with a reason without it)
```
* Reproduce the source checks yourself: try a handful of delisted tickers from `av_delisted.csv` on every source named.
  Is any of them actually free and reachable? A "not obtainable" claim is easy to be wrong about.
* Holding the last price is optimistic for a bankruptcy (true terminal value is near zero). Is that a material bias for
  the 6-month objective? Is the 3-year objective more exposed?
* Skipping re-used tickers removes legitimate history for a company whose symbol was later re-issued. How many tickers
  does the delisted list say are re-used, and what do we lose?
* Mutation record: five ingest mutants were applied (re-used ticker merged, terminal file not written, survivors
  overwritten, builder ignoring terminal, missing-date rows kept). Find a sixth that survives.

### CLAIM 21 (HIGH) — "monthly bottom, then a daily entry with a stop" shows no edge as a mechanical rule

Asserted: `scripts/study_turnaround.py`, run once and pre-registered in `setup_engine/registry/trials.jsonl` (ledger 371 to 384),
tested a Finviz-style rule over 11,278 survivor stocks since 2001. Primary: 1,568 trades, mean +0.096R net of 1% cost,
month-clustered interval [-0.112, +0.320]; excess over a date-and-volatility-matched control +0.048R [-0.083, +0.181]; the trigger
adds +0.009R over bottom-state stocks without it. Without the 10 biggest trades the mean is -0.002R; without 2020, -0.002R.
Verdict by the pre-registered rule: "no demonstrated edge". Write-up: `docs/TURNAROUND_STUDY_2026-10-07.md`.

```bash
python -m pytest tests/test_setup_engine tests/test_study -q       # 306 tests
python scripts/study_turnaround.py --limit-batches 2 --no-ledger --out /tmp/smoke.json   # NAUT signal 2026-09-29, +1.65R
python scripts/run_screen.py --as-of 2026-09-29 --prior-rules "2-Year High: 70% or more below; Monthly Low: Higher low" \
   --rules "20-Day High: New High; Relative Volume: Over 2; Price: Over $1; Dollar Volume 20d: Over 1M"
```
**A "no edge" result is easy to produce by accident (a bug that never fires, a control that is too good), so try to make the
study find an edge that is there, and try to make a null study look like an edge.**
* **Pre-registration integrity.** The constants at the top of the runner were fixed before the run, except one change made after
  the brief gave ZERO signals on the user's own example: BOTTOM is read on the previous bar (`bottom_lag=1`). Is that a
  legitimate correction or a forking path? The original reading is stability config `bottom_same_bar` (mean +0.113R, excess
  +0.072R [-0.078, +0.228]). Does the conclusion change under it?
* **Control fairness.** C3v matches on date and ATR% bucket (ATR measured the day BEFORE the signal). C3 (date only) shows a
  "significant" +0.146R. Which is the right comparison, and is the C3v bucket matching itself biased (the pool includes the
  treatment trade; bucket edges are fixed at 2/3/4/6/8/12%)?
* **Look-ahead.** Check `setup_engine/screen.py::indicators` (truncation-invariance is tested) and the order model in
  `setup_engine/trade_sim.py` (entry at next open; stop-first; trail uses the previous bar's high and ATR; gap fills at the
  open). A second implementation matched on 3,754 trades, but both follow the same written spec: is the SPEC wrong anywhere?
* **Fill optimism.** The trailing stop fills at its computed price inside the bar. The AGIG trade (+20.4R) filled at 108.45
  in a bar that ranged 166 to 59. How much of the result depends on fills a halted or illiquid stock would not give? Re-run
  with a fill at the worse of stop and the open, or with the largest 20 trades removed.
* **Concentration.** 10 trades hold 153.5R of 150.0R total; 2020 holds 152.6R. Is "no edge" or "an edge that lives in rare
  rebounds" the fairer description? Does a heavy-tailed bootstrap (not clustered ratio of sums) change the interval?
* **Survivorship.** `setup_engine/trade_stats.py::delisted_share_by_year` gives 14.1% (2013+) and 17.8% (all years) as a LOWER
  bound; break-even is 6.9-8.7%. Is the "all missing signals lose 1.0-1.3R" assumption defensible? The report also shows +0.06R
  if they behave like ordinary breakouts. `NON_COMMON_RE` also drops real names containing "Preferred" (e.g. a bank). Quantify.
* **Verdict rule.** Four conditions, all required (net E above 0; excess over C3v above 0 under month AND half-year intervals;
  excess over C2 above 0; survivorship-adjusted E above 0). Is it too strict, too loose, or biased toward "no edge"?
* **Worker claims.** Mutation tables were produced by the Workers (screen 9/9, trade_sim 11/11, trade_stats 7/7, runner 7/7,
  CLI 3/3 caught, plus extras). Re-run at least five mutants of your own choosing and report any that survive.

### CLAIM 22 (HIGH) — the 2022 chip/AI winners were unremarkable at their low, and no scanner of ours caught the bottom

Asserted (`docs/MTF_STUDY_2026-10-09.md` sections 1-2; `docs/mtf_case_study/results.json`, `scanner_recall.json`): 43 winners chosen ex post;
11 bottomed the same week (2022-10-14); at the low the median chip/AI winner sat at the 14th-21st percentile of all liquid stocks on 13/26-week
return, distance below the 30-week average, depth off the 52-week high, RS and RSI. The first flag of every scanner (Finder signal, textbook trend
setups, turnaround rule, learned capitulation rules, the new MTF signatures) came after the low, a median 15-30 weeks later at +40% to +70%, with hundreds
to thousands of other stocks flagged the same week.
```bash
python scripts/mtf_case_study.py --limit-batches 12 --workers 8 --out-dir /tmp/case_check     # percentiles need >=200 liquid stocks/week
python scripts/scanner_recall.py --limit-batches 3 --workers 8 --out-dir /tmp/recall_check
```
* The cohort is the user's names plus a hand-written chip/AI list: is any name missing that would change the "unremarkable at the low" finding (e.g. ones that were strong at the low and still ran)?
* Percentiles are same-week cross-sections among liquid stocks (price >= $2, 13-week median dollar volume >= $5M). Do they change with a different liquidity floor?
* The low is the lowest weekly close in the best trough-to-peak since 2021-10-08: an ex-post label. Check `mtf.best_run` and that it never feeds a feature.
* Scanner recall counts a flag only if it falls in [low - 8w, low + 52w]. Recompute the first-flag week for three names by hand from raw bars.

### CLAIM 23 (HIGH) — pre-registered multi-timeframe confluence signatures show no edge on held-out events

Asserted: `scripts/mtf_event_study.py` (run once, ledger 384 to 400): on 10 held-out mature SPY confirmation events, S1-S4 vs date-matched controls with the
same 30% trailing exit: excess -0.008 [-0.070, +0.061], -0.040 [-0.178, +0.185], +0.035 [-0.085, +0.169], +0.031 [-0.091, +0.195] (98.75%, event-clustered);
placebo weeks show no systematic excess; 2022 (in-sample) S3 +0.22, S4 +0.41. Verdict: no demonstrated edge.
```bash
python -m pytest tests/test_setup_engine tests/test_study -q
python scripts/mtf_event_study.py --limit-batches 3 --workers 8 --no-ledger --n-placebo 5 --out /tmp/event_smoke.json
```
* Look-ahead: weekly/monthly bars are used only once complete (calendar-label rule); the cache ends 2026-10-06 while benchmark files run to 2026-10-09 and are clipped. Find a leak.
* The control includes the signature's own rows (material for B0 and S1). Does excluding them change any sign?
* Event definition: weekly close back above the 30-week average after >= 8 weeks below with a >= 15% drawdown. 13 SPY events fall in ~3 regimes. Is the verdict fragile to 6 or 12 weeks, or to a 10% drawdown?
* Signatures were written after viewing 2022; 2022 is excluded. Is any held-out event contaminated by the exploration (2025 lows were also seen; the event is immature and excluded)?
* The harsh survivorship check assumes the signature's missing share is 1.5x the market's at -100%. All signatures already fail the interval test before it. Confirm independently.
* Independent reproduction: from `~/.roboalgo/mtf/event_rows.parquet`, recompute B0/S1-S4 trade counts, means and excess with plain pandas (the orchestrator did and matched exactly).

### CLAIM 24 (MEDIUM) — ride-exit behaviour: moving-average exits whipsaw; a 30% trail captures ~17% of the run

Asserted: from the first confluence (weekly close above a rising 30-week average and monthly close above the 10-month average), the median of 37 chip/AI winners returned
-6% (30-week exit), +2% (10-month exit) and +46% (30% trail), capture 10% / 14% / 17% of the run. Group level (SMH, 17 events): 30% trail positive 12/17 (10/10 since 2009, 2/7 before),
the moving-average exits positive 2/17 and 3/17. The group-level table is descriptive and was added after the event study.
* `setup_engine/ride.py` fills close-based exits at the NEXT open and checks the stop first; a loop-based reference in the tests agrees on 480 rides. Find a case where the order model flatters an exit.
* Entry at the first close above a moving average and an exit at the same average guarantees whipsaws: is "exposure-confounded" the right label, or is it simply a bad exit pairing?
* The 30% trail sold AMD at +45% in mid-2024 (it went on to x11): is a fixed trail the right primary exit, and was it chosen before seeing results? (It was fixed in the plan before any run.)

### CLAIM 25 (HIGH) — "today's blue chips" flatter a backtest by about 2 points per half-year, and chart setups add little in blue chips

Asserted (`docs/BLUECHIP_STUDY_2026-10-12.md`, scripts in `scripts/exploratory/`): today's S&P 500 members averaged +6.0%/+8.6%/+6.7% per half-year vs +3.9%/+6.4%/+4.9% for the top 500 by dollar volume at the time;
trend setups add +0.3% to +1.8% per half-year there; SPY above its 30-week average is not an entry signal.
```bash
python scripts/exploratory/bluechip_trend.py ; python scripts/exploratory/bluechip_regime.py ; python scripts/exploratory/bluechip_pullback.py ; python scripts/exploratory/bluechip_sector.py
```
* Join dates come from the Wikipedia table, where early rows carry placeholder dates (many 1957 or 1976). Does that weaken the join-date-aware universe?
* "Top 500 by dollar volume" uses the 50-day dollar volume in the observation set (survivors only): is it a fair point-in-time proxy for blue chips? What does it miss (Lehman, Wachovia, Countrywide)?
* About ten tests were run with no multiplicity correction; the RS-leader, pullback and sector rows were exploratory. Which conclusions survive a Bonferroni level?

### CLAIM 26 (HIGH) — 50x stocks are 0.07% of start points; the fallen low-priced profile is a long shot; waiting for trend confirmation lowers the odds

Asserted (`docs/MULTIBAGGER_STUDY_2026-10-12.md`, `scripts/multibagger_study.py`, SURVIVORS ONLY): 12 distinct stocks reached 50x within 10 years of a December start 2001-2015; the fallen $1-$10 profile hit 50x 0.84% of the time with 31.8% falling another 50%.
```bash
python -m pytest tests/test_study/test_multibagger_study.py -q ; python scripts/multibagger_study.py --workers 8
```
* Overlapping monthly starts inflate n: is the distinct-stock count the honest sample size? Recompute the 50x share per distinct stock.
* The forward window excludes the start month and requires 120 full months; a stock delisted inside the window is missing in survivors-only runs. Quantify the bias with the delisted list once prices are merged (Tiingo loader).
* Is the 60-month-high drawdown distorted by reverse splits? (36 of 521 pool names carry suspected unadjusted splits.)

### CLAIM 27 (MEDIUM) — unadjusted splits and spin-offs do not change the earlier studies

Asserted (`docs/ARTIFACT_SENSITIVITY_2026-10-12.md`): dropping contaminated trades moves the turnaround primary mean from +0.0957R to +0.0902R and the event-study excess by about 0.0003; BEFORE reproduces the published numbers exactly.
```bash
python -m pytest tests/test_setup_engine/test_artifacts.py tests/test_study/test_artifact_sensitivity.py -q ; python scripts/artifact_sensitivity.py --workers 8 --out /tmp/art.json
```
* The detector is post-hoc. Design a case it misses (artifact with a volume spike) and one it wrongly flags (halt-reopen); quantify both on the flagged list in the JSON.
* The control is deliberately left uncleaned; does cleaning it too change any sign?

### CLAIM 28 (MEDIUM) — the Tiingo loader and the SEC provider are correct offline; their live behaviour is untested

Asserted: `scripts/tiingo_load_delisted.py` / `data_engine/tiingo_provider.py` (split-adjust, identity rule, throttle, token never persisted) and `data_engine/sec_edgar.py` (point-in-time rule on the `filed` date, quarterly differencing, runway) pass 171 tests with all planned mutants caught.
```bash
python -m pytest tests/test_data_engine/test_tiingo_provider.py tests/test_ingest/test_tiingo_load.py tests/test_data_engine/test_sec_edgar.py -q
python scripts/tiingo_load_delisted.py --dry-run
```
* Tiingo `splitFactor` semantics were inferred from two live examples (a 2:1 in 1993 and a 1:4 reverse split in 2016); the adjusted series was not compared to `adjClose` live. Design that comparison.
* XBRL pitfalls: amended filings duplicate facts; comparatives re-reported in later filings carry the later filed date; revenue concepts change around 2018; 52/53-week years; debt omits convertibles and leases. Which produces silently wrong numbers?
* Confirm no credential or e-mail is present in the code, tests or fixtures (the only address-shaped strings should be reserved examples).

---

# PART 2 — INDEPENDENT REVIEW (no claims supplied; find what was missed)

1. **Run the suite.** Baseline before this work was **203 passed / 9 failed**, all 9 in
   `tests/test_trade_signal_engine/` and pre-existing. After this work it should be
   **327 passed / 9 failed** (203 + 17 TradingView + 47 validation gates + 7 database-URL + 28 Finder replay + 16 entry-status + 9 incomplete-bar), on BOTH Python 3.14 and a Python 3.11 built from `requirements.txt`, with the same 9
   failures and **zero failures outside that directory**. If the 9 now pass, something
   changed that was not asked to change — investigate.
   ```bash
   /Users/asmian/RoboAlgo/venv/bin/python -m pytest tests/ -q
   cd frontend && npx tsc --noEmit && npm run build
   ```
   **Note a material caveat the audit flagged about its own results:** the local venv is
   **Python 3.14.8** while `render.yaml` pins **3.11** on all three services, with no
   `runtime.txt` to reconcile them. Every test result above was produced on 3.14. A green
   local suite does not prove a green production deploy. Judge whether that skew invalidates
   any specific claim here.

2. **Read the full diff.** `git diff` and `git status`. For every changed file ask: does this
   change the numbers a trader sees? Is there a path where it fails silently?

3. **Hunt specifically for the failure mode this codebase keeps producing: silent empties
   and swallowed exceptions.** Grep for `except: pass`, bare `except Exception`, `.get(x, 0)`
   where the key can be present-but-NaN, and functions returning an empty DataFrame/list on
   error. `int(NaN)` raises; `float(NaN) or 0` returns NaN because NaN is truthy. Both have
   already caused production bugs here.
   ```bash
   grep -rn "except.*:\s*pass\|except Exception" --include=*.py . | grep -v venv | grep -v tests/
   ```

4. **Concurrency and blocking.** uvicorn runs ONE worker. Any multi-second handler freezes
   the entire API — this already stalled `--reload` mid-request and looked like a frontend
   bug. These routes were measured: `/api/finder/scan/wide`, `/api/patterns/scan`,
   `/api/trade-signals/scan` all >25s; `/api/analytics/data-quality` 17.9s;
   `/api/crypto-signals/scan` 15.5s. Is any of them declared `async def`? That would move a
   blocking call onto the event loop and make it worse. Check.

5. **Known-broken routes.** These returned errors in a smoke test of all 83 parameterless GET
   routes: 500 on `/api/analytics/expectancy`, `/api/analytics/regime-performance`,
   `/api/analytics/setup-performance`, `/api/rocket/scan`; 422 on
   `/api/analytics/regime-timeline`; connection death on `/api/trade-signals/scan/stream`.
   Were they fixed, documented, or ignored? Is a 500 on a live route acceptable here?

6. **Secrets.** A committed AlphaVantage key exists in `settings.json` and in git history
   (`ddf19f2`) and needs rotation — that is the operator's decision, not a code fix. Confirm
   no new secret value was written into any file, and that no key appears in the new docs.
   ```bash
   git diff | grep -iE "api[_-]?key|secret|token" | grep -vE "sync: false|getenv|ALPHAVANTAGE_API_KEY=|os.environ"
   ```

7. **Legal/ToS — report, do not decide.** TradingView's Terms of Service prohibit scraping
   and redistribution. `scanner.tradingview.com/america/scan` is undocumented and can be
   rate-limited or removed without notice. Confirm the integration is (a) one provider behind
   a working fallback, never the sole source, (b) rate-limited and cached, (c) not
   redistributing data. Then state the residual risk plainly and leave the decision to the
   operator.

8. **The thing nobody asked about.** You have the whole repo. What is the most dangerous
   thing in it that is *not* mentioned anywhere above? Name one, with evidence.

---

# PART 3 — OUTPUT FORMAT

```
## Verdict
<2-3 sentences: is this work safe to keep? What is the single biggest risk?>

## Claim audit
| # | Claim | CONFIRMED / REFUTED / UNVERIFIABLE | Evidence (actual command output) |

## Defects found (severity-ordered)
| Severity | File:line | Defect | How it fails in practice | Suggested fix |
Severity = CRITICAL (wrong trade / silent data loss) > HIGH > MEDIUM > LOW.

## Overstated or unverified claims
<Anything asserted more confidently than the evidence supports. Be specific.>

## What I would do next, in order
```

**Rules for your review:**
- Every finding cites a file:line or a command with its real output. No finding from reading
  a summary alone.
- If you cannot verify something, say **UNVERIFIABLE** and why. Do not guess, and do not
  pad the report to look thorough.
- Prefer one proven CRITICAL over ten speculative MEDIUMs.
- If the work is sound, say so plainly — but only after trying to break it.
- You are reviewing a system that can place real trades. **A wrong number is worse than a
  missing feature.**
