# Data sources runbook (2026-10-12)

What each data piece is, how to run it, and what to expect. No credentials are stored in the repo; tokens come from environment variables.

## Daily blue-chip screen (works now, free)

```bash
python scripts/bluechip_daily.py                      # S&P 500 list (refreshed weekly), fresh prices, screen, fundamentals snapshot
python scripts/bluechip_daily.py --no-fundamentals    # skip the Yahoo snapshot
python scripts/bluechip_daily.py --limit 25 --no-fundamentals    # smoke
```
Writes `~/.roboalgo/reports/bluechip_<as_of>.{csv,md,json}`. The weekly state changes only on Friday closes; the report says which week it reflects. Chart state is **context, not a signal**: see `docs/BLUECHIP_STUDY_2026-10-12.md`. The S&P 500 list is parsed from Wikipedia by `setup_engine/universe.py` and cached in `~/.roboalgo/universe/`.

## Tiingo: delisted-stock prices (needs the paid plan to finish)

Verified with a real key on 2026-10-12: 37 of 40 sampled delisted stocks (2010-2025) have complete data under the right company; 3 were re-used tickers. Bankrupt names live under a Q ticker. Gap: pre-2010 delistings are mostly absent.

```bash
export TIINGO_TOKEN=...                                           # never written to disk by the loader
python scripts/tiingo_load_delisted.py --dry-run                  # counts and time estimate, no network
python scripts/tiingo_load_delisted.py --plan free --max-requests 20 --limit-symbols 8     # live smoke; read the state reasons
python scripts/tiingo_load_delisted.py --plan power               # full run, resumable (state in ~/.roboalgo/tiingo/state.json)
python scripts/ingest_delisted_history.py --source ~/.roboalgo/tiingo/out
python scripts/build_setup_dataset.py --history ~/.roboalgo/history_all
```
* **Size:** 3,213 delisted common stocks to load. Free plan (50 requests an hour, 500 symbols a month): about 400 days. **Power plan ($30 for one month): about 1 hour.** Free-plan sampling is the only no-cost route.
* **Prices** are raw; the loader converts to split-adjusted (divides by the cumulative later split factors, scales volume), the same convention as the survivor cache; dividends are not adjusted.
* **Identity checks** reject tickers re-used by another company; the state file records why each symbol was skipped.
* **After loading:** run `scripts/artifact_sensitivity.py` logic (`setup_engine/artifacts.py`) on the merged history before trusting any delisted-inclusive number, then rerun `scripts/multibagger_study.py --history ~/.roboalgo/history_all`.
* **License:** Tiingo's plans are "internal use only"; fine for research, not for showing the data to others.

## SEC EDGAR: point-in-time fundamentals (needs a contact for the User-Agent)

```bash
export SEC_USER_AGENT="Your Name contact@your-domain.example"     # REQUIRED: the SEC returns 403 without a contact
```
`data_engine/sec_edgar.py` provides ticker-to-CIK, company facts as a point-in-time table (a value is visible only from its `filed` date; restatements later than the as-of date are invisible), quarterly values differenced from year-to-date figures, trailing-twelve-month sums, a snapshot (cash, debt, operating cash flow, burn, runway, shares, share growth, revenue growth, gross margin) and recent filings with dilution flags (S-3 plus 424B5 is the meaningful at-the-market signal; 424B3 is mostly resale or merger). It never uses the SEC "frames" endpoint, which reflects later restatements. Throttled to 8 requests a second, cached on disk. Tested offline with synthetic fixtures only; the first live run has not happened.

## Other vendors checked

* **Market Data (marketdata.app):** the account tested is Free/Trial with one year of history (AAPL exempt); not used.
* **Alpha Vantage delisted list** (`~/.roboalgo/delisted/av_delisted.csv`): 604 of 9,550 rows carry the snapshot date 2026-10-06 instead of a real delisting date; the loader excludes them.

## Security note
Tokens pasted into a chat session are in that transcript. Regenerate the Tiingo and Market Data tokens after use, and rotate the old AlphaVantage key that is still in git history.

## Alpaca (market data; account and orders live in broker_engine/alpaca.py)

Not yet run with a real key. `data_engine/alpaca_provider.py` is a read-only market-data client (plain HTTP, no SDK): daily or intraday bars, the asset list (active or inactive, with the exchange so OTC rows can be told apart), the market clock and corporate-action announcements. It places no orders. Trading stays in `broker_engine/alpaca.py`, which uses the alpaca-py SDK.

```bash
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...         # same names as broker_engine/alpaca.py; paper-account keys are enough; never written to disk
python scripts/alpaca_check.py                           # go/no-go: auth, AAPL since 2016, delisted-coverage probe (40 stocks, seed 7)
python scripts/alpaca_check.py --plan plus --sample 100  # paid plan: faster throttle, bigger sample
```
* **Verdict:** PASS when at least half of the sampled delisted common stocks (AlphaVantage list, delisted 2017-2025) have bars from listing, or from 2016-01-04 for older listings, to within 45 days of delisting. FAIL means keep using the Tiingo loader. The report goes to `~/.roboalgo/reports/alpaca_check_<date>.json`. Read the per-stock rows: a re-used ticker can pass the date test.
* **Plans:** free (Basic): 200 requests a minute, IEX in real time, SIP history without the latest 15 minutes (the client trims `end` on the free plan). Algo Trader Plus: 10,000 requests a minute. History starts in 2016, so delistings before 2016 are out of reach.
* **Feed:** the client defaults to `iex` (IEX-exchange trades only). Studies should pass `feed="sip"`; the check script does.
* **Adjustment:** always sent explicitly (`split` by default, matching the survivor cache). Run the artifact detector on delisted prices before trusting them.
* **Unverified:** the OAuth route (ALPACA_CLIENT_ID and ALPACA_CLIENT_SECRET exchanged at authx.alpaca.markets, used only when ALPACA_API_KEY is not set; it may belong to Alpaca's Broker API rather than personal accounts), the `asof` symbol-mapping parameter, and the corporate-actions path `/v2/corporate_actions/announcements` (90-day windows, data since April 2020, no delistings or symbol changes).
* **Host:** `ALPACA_PAPER=false` moves the asset and clock calls to api.alpaca.markets (read-only). Anything else uses the paper host.
