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
