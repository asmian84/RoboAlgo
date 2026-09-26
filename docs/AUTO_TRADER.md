# Auto-Trader Operator Runbook

This document is for whoever is on call for the autonomous trading loop. It
assumes you have not read `auto_trader/*.py` and need to operate the system
safely anyway. Where this doc and the code disagree, the code wins — but if
that happens, treat it as a bug in this doc and fix the doc.

Source modules this runbook is based on: `auto_trader/config.py`,
`auto_trader/governor.py`, `auto_trader/state.py`, `auto_trader/audit.py`,
`auto_trader/reconciler.py`, `auto_trader/loop.py`,
`api/routers/trading_control.py`, `infrastructure/celery_app.py`,
`infrastructure/scheduler.py`, `scripts/auto_trader_smoke.py`,
`scripts/migrate_auto_trader.py`, `render.yaml`.

---

## 1. What it does

Celery beat fires the `auto_trader_cycle` task every 15 minutes, Monday
through Friday, 13:00–20:00 UTC (US regular market hours), on the `trading`
queue (`infrastructure/scheduler.py`). Each firing:

1. Acquires a Redis cycle lock (`state.acquire_cycle_lock`, 900s TTL) so two
   overlapping workers can never run the same cycle twice. If the lock is
   already held, the task returns `{"status": "locked"}` and does nothing
   else.
2. Runs `auto_trader.loop.run_cycle()`, which:
   - Reconciles against the broker first — the broker's account, positions,
     and open orders are read fresh and treated as ground truth for
     everything downstream (`reconciler.reconcile`).
   - Runs account-wide halt checks (kill switch, daily loss limit,
     consecutive-loss halt).
   - Scans a symbol universe for BUY/SELL candidates (equities via
     `trade_signal_engine.scanner.scan_universe`, or crypto via
     `trade_signal_engine.crypto_scanner.scan_crypto_universe` if
     `asset_class="crypto"`), keeping the top `max_new_positions_per_cycle`
     candidates by score.
   - Risk-gates every candidate through the governor (see section 5) before
     touching a broker.
   - Places orders that clear the governor as **bracket orders**: entry +
     stop-loss + take-profit submitted to the broker in one call
     (`broker.place_bracket_order(...)`). Because the stop and target live at
     the broker, not in this process's memory, a crashed worker or a dead
     server does not leave a position unprotected — the broker will still
     execute the stop.
3. Releases the cycle lock when done, regardless of outcome.

`max_retries=0` and `acks_late=False` on the Celery task are deliberate: if
this task were acked late (the Celery default), a worker crash mid-cycle
would cause the broker (Redis) to redeliver the task to another worker, which
would re-run the cycle and potentially double-place orders. Combined with the
Redis cycle lock and the deterministic `client_order_id` generated per
`(cycle_id, symbol, side)` in `reconciler.client_order_id_for`, this is one of
three independent layers preventing a double-fill.

---

## 2. Required environment variables

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY` | Alpaca credential (worker only) |
| `ALPACA_SECRET_KEY` | Alpaca credential (worker only) |
| `ALPACA_PAPER=true` | Tells the Alpaca adapter to hit the paper endpoint. Defaults to `true` if unset — you must set it explicitly to `false` to point at a live account, and that alone is not enough (see below). |
| `BROKER=alpaca` | Selects the broker adapter (`alpaca` \| `ibkr` \| `questrade`) via `broker_engine/factory.py`. |
| `ROBOALGO_REDIS_URL` | Redis connection string, shared by Celery (broker/backend) and `auto_trader.state` (kill switch, run flag, cycle lock, heartbeat). |
| `ROBOALGO_CONTROL_TOKEN` | Bearer-style token required on risk-increasing control endpoints. If unset, those endpoints return `503` rather than running open or silently no-op'ing. |

### The two live-mode switches

Live trading requires **both**:

1. The environment variable `ROBOALGO_ALLOW_LIVE=1` on the process that runs
   the cycle (the Celery worker), and
2. `live_confirmed: true` persisted in the auto-trader config.

Both gates are enforced independently and redundantly:

- `auto_trader/config.py`'s `save()` **refuses to persist** `mode: "live"` or
  `live_confirmed: true` unless `ROBOALGO_ALLOW_LIVE=1` is already set in the
  environment — a bad request body or a UI bug cannot silently arm live
  trading through the config API.
- `auto_trader/config.py`'s `assert_mode_safe()` is called at the top of
  every cycle and raises `AutoTraderConfigError` (which fails the cycle with
  status `failed`, it does not trade) if `cfg.mode == "live"` and the live
  gate (`ROBOALGO_ALLOW_LIVE=1` AND `live_confirmed is True`) is not open —
  **regardless of what the broker itself reports**.

**`mode: "paper"` combined with a broker that reports live trading (i.e.
`broker.is_paper is False`) is always a hard failure**, independent of the
live gate. `assert_mode_safe()` raises unconditionally in that case
("config says paper, broker reports live"). There is no configuration in
which a paper-mode cycle is allowed to touch a live account. Conversely, a
broker reporting live trading also requires the live gate to be open even if
`cfg.mode` somehow said otherwise.

---

## 3. First-run checklist (in order)

Run these in order. Do not skip steps or reorder them — several of them are
silent-failure traps if done out of sequence.

1. `pip install -r requirements.txt`
2. `python scripts/migrate_auto_trader.py` — **one-time**, creates the three
   audit tables (`auto_trader_cycles`, `auto_trader_decisions`,
   `auto_trader_discrepancies`). Safe to re-run (`checkfirst=True`). If you
   skip this, the loop still runs — every `audit.*` write function swallows
   its own exceptions and logs a warning instead of raising, by design, so a
   missing audit table does not stop trading. It just means **the entire
   audit trail silently writes nothing**, and you will have no record of
   what the loop did.
3. `python scripts/auto_trader_smoke.py` — must exit `0`. This is a fully
   offline smoke test (no Alpaca creds, no Redis, no live market, no real
   DB — it monkeypatches the broker, the scanner, and the DB session) that
   proves the governor, the dry-run path, and idempotent duplicate-order
   skipping all work end to end. If this fails, do not proceed.
4. Start Redis (or point `ROBOALGO_REDIS_URL` at a running instance).
5. Start the Celery worker and beat:
   - `celery -A infrastructure.celery_app worker --loglevel=info --queues=pipeline,symbols,trading --concurrency=2`
   - `celery -A infrastructure.scheduler beat --loglevel=info`
6. Set `enabled: true` with `dry_run: true` via `POST /api/auto-trader/config`
   (token-gated).
7. Watch one full cycle go by in the AUTO page (or via `GET
   /api/auto-trader/cycles` / `GET /api/auto-trader/decisions`) and confirm
   it looks sane — right symbols, right verdicts, no unexpected `REJECT`
   spam.
8. Only then set `dry_run: false` and watch paper fills happen for real
   against the paper account.

---

## 4. How to kill it fast

Three ways, in escalation order:

1. **Kill switch via API** (fastest, no token needed):
   ```
   curl -XPOST <host>/api/auto-trader/kill \
     -H 'content-type: application/json' \
     -d '{"on":true,"reason":"..."}'
   ```
   This sets the kill switch in Redis. The loop checks it at the very start
   of every cycle, and again on every iteration of the per-candidate loop
   inside a cycle, so a long-running cycle reacts to it mid-flight rather
   than finishing first.
2. **The red KILL button on the AUTO page** — calls the same endpoint.
3. **`celery -A infrastructure.celery_app control shutdown`** — stops the
   worker process outright. This is the blunt option: it does not touch
   anything already placed at the broker (bracket orders live at the broker
   and keep working), it just stops this process from starting or continuing
   any cycle.

### Why `/stop` and `/kill {on:true}` need no token

`api/routers/trading_control.py` is explicit about an auth asymmetry: this
API has **no authentication layer anywhere else** in the codebase (auth is
frontend-only, enforced by Supabase in the browser). The auto-trader control
endpoints are the one exception, and even there the asymmetry is deliberate:

- Risk-**reducing** actions — `POST /stop` (disable the run loop) and `POST
  /kill` with `on: true` (engage the kill switch) — are **unauthenticated on
  purpose**. A missing or wrong `X-RoboAlgo-Control-Token` must never be able
  to block an operator from halting the system.
- Risk-**increasing** actions — `POST /start`, `POST /config`, `POST
  /cycle/run`, and `POST /kill` with `on: false` (turning the kill switch
  back OFF) — all require `X-RoboAlgo-Control-Token` to match
  `ROBOALGO_CONTROL_TOKEN`. If that env var isn't set at all, these endpoints
  return `503` rather than running open or silently doing nothing.

In short: you can always turn it off without credentials; you always need
credentials to turn it back on or increase what it's allowed to do.

---

## 5. Governor rules (`governor_rule` on REJECT/HALT decisions)

Evaluated in this precedence order — first match wins. Rules 1–3 are
account-wide halts checked before any candidate is even considered; rules
4–10 are per-order checks.

| Rule | Meaning | What to do |
|---|---|---|
| `kill_switch` | The kill switch is engaged. | Expected if you just killed it. If unexpected, check `GET /status` → `kill_meta` for who set it and why, and Redis reachability. |
| `daily_loss_limit` | Today's equity is down ≥ `daily_loss_limit_pct` (default 3%) from `day_start_equity`. | Trading is halted for the rest of the day by design. Do not override without understanding why the loss happened — check recent `EXIT` decisions and discrepancies first. |
| `consecutive_loss_halt` | `consecutive_loss_halt` (default 3) losing exits in a row, per `audit.consecutive_losses()`. | Same as above — this is a "something is systematically wrong" circuit breaker, not noise. Investigate before resuming. |
| `invalid_order` | The candidate itself is malformed (qty/entry/stop ≤ 0, or stop on the wrong side of entry for the direction). | This points at a bug in the scanner or position-sizing math, not risk policy. Reject is correct; go fix the signal generator. |
| `duplicate_position` | A position already exists for that symbol. | Expected — the loop won't stack a second position on top of one it already holds. No action needed. |
| `max_open_positions` | Open position count is already at `max_open_positions` (default 5). | Expected under normal operation once the book fills up. Raise the limit only if you've reviewed the risk math. |
| `max_position_size` | Candidate notional exceeds `max_position_size_pct` (default 20%) of equity. | Expected for oversized signals. Check the scanner's position sizing if this fires a lot. |
| `per_trade_risk` | Candidate's `risk_amount` (entry-to-stop distance × qty) exceeds `per_trade_risk_pct` (default 1%) of equity. | Same — a sizing/stop-distance problem upstream if this fires often. |
| `max_total_exposure` | Gross exposure + this candidate's notional would exceed `max_total_exposure_pct` (default 60%) of equity. | Expected as the book fills. Not a bug. |
| `insufficient_buying_power` | Candidate notional exceeds available cash. | Check the account's actual cash/buying power — could indicate a reconciliation lag or a broker-side issue. |

`governor_rule` also appears as `"kill_switch"` on a mid-cycle `HALT`
decision row (kill switch tripped between candidates), and as
`"broker_error"` on a `SKIP` row when `place_bracket_order` itself raised.

---

## 6. Reading the audit trail

Two tables matter day to day: `AutoTraderCycle` (one row per cycle) and
`AutoTraderDecision` (one row per symbol-level decision within a cycle).
Query them via `GET /api/auto-trader/cycles`, `GET
/api/auto-trader/cycles/{cycle_id}/decisions`, or `GET
/api/auto-trader/decisions`.

### Cycle statuses

| Status | Meaning |
|---|---|
| `completed` | Ran to the end normally (may have placed 0 or more orders). |
| `halted` | An account-wide halt (kill switch, daily loss, consecutive losses) fired either before scanning or mid-way through the candidate loop. Check `halt_reason` for which rule. |
| `skipped` | Market was closed (and `ignore_market_hours` was false), so the cycle did nothing but log that fact. |
| `failed` | An unhandled exception was caught (config unsafe, scanner error, or an uncaught crash anywhere in `run_cycle`) — see `error` on the cycle row / the traceback it stores. |
| `disabled` | `cfg.enabled` is `False`. Returned before any I/O or DB write — there is no cycle row for this one at all. |
| `killed` | Kill switch was on at the very start of the cycle, before reconciliation even ran. No cycle row (short-circuits earliest, same as `disabled`). |
| `stopped` | The run flag (`state.running_enabled()`) was off. No cycle row. |
| `locked` | Another worker already held the Redis cycle lock for this `cycle_id` (Celery-level double-dispatch protection). No cycle row from this invocation. |

Note that `disabled`, `killed`, `stopped`, and `locked` never reach the
database — they're the earliest-possible short-circuits, before
`audit.start_cycle` is ever called. If you need to know "why didn't a cycle
run at all," you have to check `GET /status` (`enabled`, `kill_switch`,
`running`) rather than the audit trail, because there's nothing in the audit
trail to look at.

### Decision actions

| Action | Meaning |
|---|---|
| `PLACE` | Bracket order was successfully submitted to the broker. |
| `REJECT` | Governor rejected the candidate (see `governor_rule`). |
| `SKIP` | Not rejected by risk policy, but not placed either — duplicate `client_order_id` already seen, or the broker call itself raised (`governor_rule: broker_error`). |
| `DRY_RUN` | Cleared the governor and the duplicate check, but `dry_run` is enabled, so nothing was sent to the broker. |
| `EXIT` | A position that `auto_trader` believed was open vanished from the broker between cycles (closed externally, or a bracket stop/target fired). Logged by the reconciler, not the candidate loop. |
| `NO_ACTION` | Explicitly-logged inaction — e.g. "market closed," "0 BUY/SELL verdicts," or "redis unavailable — forced dry_run." |
| `HALT` | An account-wide halt fired, either before the scan or mid-candidate-loop. |

**Inaction is logged too.** A cycle that scans 30 symbols and finds nothing
tradeable still writes a `NO_ACTION` decision row and a `completed` cycle
row with `candidates: 0`. Don't read "no decision rows" as "the loop didn't
run" without also checking the cycle row's status.

---

## 7. Known limitations — read before trusting this system with money

- **`realized_pnl` on `EXIT` rows is an estimate, not a fact.** The loop
  never observes a bracket exit firing at the broker in real time — it only
  notices the position is gone on the *next* cycle's reconciliation, and logs
  the last-known `unrealized_pnl` from before the position vanished as the
  "realized" P&L (`reconciler.reconcile`, `closed_since_last`). This is good
  enough to drive `consecutive_loss_halt` (a rough "are we on a losing
  streak" signal) but it is **not accounting-grade**. Do not use these
  numbers for tax reporting, performance attribution, or anything that needs
  to be exactly right — reconcile against the broker's own fill/transaction
  history for that.
- **Without Redis, state falls back to a file on one host, and the loop
  refuses to trade for real.** `auto_trader.state` fails safe: if Redis is
  unreachable, `kill_switch_on()` returns `True` (assume killed) unless
  `ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS=1` is set, in which case it falls back
  to a JSON file on the local host. `loop.run_cycle` additionally
  force-downgrades any non-dry-run cycle to `dry_run` whenever
  `state.redis_available()` is `False`, because the kill switch cannot be
  trusted to be cross-process without Redis. **`ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS=1`
  is for local, single-host testing only — never set it in a multi-host
  deployment.** In a multi-host deployment, a file-backed kill switch on one
  host is invisible to the others, which defeats the entire point of the
  switch.
- **The API has no authentication layer beyond the control-token guard on
  risk-increasing auto-trader endpoints.** Every other endpoint in this repo
  is open — auth is enforced only in the browser frontend (Supabase), not on
  the API itself. Do not assume any endpoint other than
  `/start`, `/config`, `/cycle/run`, and `/kill {on:false}` requires
  credentials.
- **`GET /status` reports `is_paper: null` in production.** The API service
  deliberately holds no broker credentials (see `render.yaml` — the API
  service is never given `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`), so it cannot
  ask the broker whether it's paper or live and degrades to `null` with a
  `broker_error`. The actual paper-vs-live enforcement happens in the worker
  process, in `config.assert_mode_safe()`, not in anything `/status`
  displays. Don't treat `/status`'s `is_paper` field as a safety indicator —
  it's informational at best and absent in the normal production topology.
- **Nothing in this system has demonstrated positive expectancy.** Nothing
  here — the scanner, the risk model, the governor thresholds — has been
  shown to make money over any real sample. Paper mode exists precisely to
  establish (or disprove) that before any live capital is at risk. Treat
  every paper-mode result as an open question, not a track record, until
  there's a real scorecard behind it.

---

## 8. Going live

Live trading requires setting **both** switches described in section 2:

1. `ROBOALGO_ALLOW_LIVE=1` in the Celery worker's environment.
2. `live_confirmed: true` saved into the auto-trader config (via `POST
   /api/auto-trader/config`, token-gated) — which itself is only accepted
   by `config.save()` once switch 1 above is already set.

Pre-flight checklist before flipping either switch:

- [ ] Paper mode has run long enough, with `dry_run: false`, to produce a
      meaningful sample of real paper fills — not just a clean smoke test.
- [ ] You have reviewed recent `AutoTraderDecision` rows for `REJECT`
      patterns that suggest the risk limits or position sizing need
      adjustment before real money is involved.
- [ ] You have reviewed `AutoTraderDiscrepancy` rows — recurring
      `broker_read_error` or `position_unknown_locally` entries mean the
      reconciler is regularly out of sync with the broker, which you want
      solved in paper mode, not live.
- [ ] `ALPACA_PAPER` is explicitly set to `false` on the worker (and
      `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` point at the live account, not the
      paper one) — remember `ALPACA_PAPER` defaults to `true` if unset, so
      this must be an explicit, deliberate change, not an omission.
- [ ] `ROBOALGO_REDIS_URL` points at a real, reachable Redis instance you
      trust to stay up — with Redis down, the loop force-downgrades to
      dry-run (section 7), so a spotty Redis silently turns "live" back into
      "paper" rather than failing loudly.
- [ ] You have a kill-switch plan ready (section 4) and have tested that the
      unauthenticated kill call actually reaches this deployment before you
      need it under pressure.
- [ ] `mode: "live"` is set in config alongside `live_confirmed: true` —
      recall that `mode: "paper"` with a broker that reports live trading is
      an unconditional hard failure (`assert_mode_safe` raises), so mode and
      the broker's actual account must agree before the switches will do
      anything but fail the cycle.

Once both switches are set and the config round-trips successfully (the
`POST /config` call would otherwise raise `400 AutoTraderConfigError`), the
next scheduled cycle — or a manually triggered `POST
/api/auto-trader/cycle/run` — will place real orders against the live
account, gated by the same governor rules as paper mode.
