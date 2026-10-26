# External repository evaluation

Evaluated 2026-10-06. Question asked: "see if these repos can make us better."
All five repos were reachable and evaluated. Evidence was fetched from the GitHub REST API
(unauthenticated; `gh` was not logged in), raw README/LICENSE/source files, and the arXiv HTML
for TradingAgents. Nothing was installed, no code was changed, and no repo was cloned or run.
Things I did NOT verify are listed per repo and in the last section.

## Baseline: what we already have

- We call TradingView's screener directly: `POST scanner.tradingview.com/america/scan`, no key,
  in `data_engine/tradingview_provider.py`. That file deliberately requests only `name`, `close`,
  `volume`, because TradingView's precomputed indicator columns are not corporate-action-safe.
  Its docstring records NYSE:CTVA at close 12.39 with SMA50 77.32, SMA200 77.88, RSI 6.38, change
  +3.94%, while SPY/MSFT/NVDA/AAPL look fine. The corruption is silent and symbol-dependent.
- Strategy validation with a Deflated Sharpe gate and mandatory OOS: `strategy_validation/gates.py`.
- Rule-based personas, no LLM: `lab_engine/personas.py`. Verdict log: `lab_engine/verdict_log.py`.
- Frontend charts use `lightweight-charts` and `klinecharts` (`frontend/package.json`).
- Our venv reports Python 3.14.8 and pandas 2.3.3 (the brief said 3.11; the venv is what I checked).

## Summary

| Repo | License | Maintained (as of 2026-10-06) | Language | Verdict |
|---|---|---|---|---|
| atilaahmettaner/tradingview-mcp | MIT (LICENSE file) | Yes, last push 2026-10-04 | Python | REFERENCE-ONLY |
| Open-Dev-Society/OpenStock | AGPL-3.0 (LICENSE file) | Yes, last push 2026-10-01 | TypeScript (Next.js) | SKIP |
| Mathieu2301/Tradingview-API | No LICENSE file; `package.json` says ISC; GitHub detects none | Burst of activity 2026-10-02 to 10-05; stable v3.5.2 is from 2025-10-05 | TypeScript (Node >= 20) | SKIP |
| TauricResearch/TradingAgents | Apache-2.0 (LICENSE file) | Yes, v0.6.0 on 2026-10-03 | Python >= 3.11 | BORROW-IDEA (minor) |
| tradesdontlie/tradingview-mcp | MIT plus appended notices (GitHub shows "Other/NOASSERTION") | No commits since 2026-07-28 (70 days) | JavaScript (Node 18+) | SKIP |

None of the five is worth adopting as a dependency. Details and evidence follow.

## 1. atilaahmettaner/tradingview-mcp

- **What it does.** An MCP server (37 tools per README) wrapping the `tradingview-screener` and
  `tradingview-ta` PyPI packages. I confirmed in `screener_provider.py` that it hits
  `https://scanner.tradingview.com/{market}/scan`, the same endpoint we already call. Backtests use
  Yahoo's chart API with `adjclose` scaling (`backtest_service.py`), not TradingView data.
- **License.** MIT, LICENSE file present. Passes the gate.
- **Maintained.** Yes: last push 2026-10-04, active PRs. It also has a paid hosted tier
  (README: $9/mo and $29/mo), which is the maintainer's business model.
- **Runtime cost.** Python, but `pyproject.toml` says `requires-python >=3.10,<3.14`, so it would not
  install into our 3.14.8 venv as-is. It pins `tradingview-screener==3.0.0` because, per its own
  comment, 3.2.0 "silently returns 0 rows for crypto/futures scans".
- **Credentials.** None required. Optional free Marketaux token for news.
- **Marginal value over what we have: almost none.**
  - Non-US markets: it passes a market path segment to the same endpoint. We could do that by
    changing the URL constant in our own provider. I did not probe non-US markets live.
  - Its "multi-agent debate" is three rule sets. The repo itself renamed it "Rule-Based Signal Summary"
    and says "This is not an AI debate" (`multi_agent_service.py`). It matches our persona decision.
  - Walk-forward: its own caveat says no parameters are optimised, so it "measures regime consistency
    ... not parameter overfitting". `compare_strategies` ranks 9 strategies on one symbol; in the files
    I read (`backtest_service.py`) I found no multiple-testing correction. Our `strategy_validation/gates.py`
    is stricter. README examples such as "+31.5% return | 100% win rate | 2 trades" are 2-trade results.
- **Indicator hazard: YES, flagged.** `multi_agent_service.py` feeds TradingView's precomputed `RSI`,
  `MACD.macd`, `SMA20`, `EMA200`, `close` and `change` into BUY/SELL decisions. That is exactly the
  CTVA failure mode. I did not run it against CTVA, so I cannot say what it outputs there.
  I read the multi-agent file and the README tool list, not every tool.
- **Silent-failure history.** Recent commit "fix: correct wrong exchange codes in FUTURES_WATCHLIST
  (ES/NQ/RTY/YM silently return 0 rows)". The wrapper stack has shipped silent-empty bugs.
- **ToS.** Same scanner endpoint as ours, but an MCP server (and the hosted tier) that relays TV data to
  third parties is a larger redistribution exposure than our private in-process call. README disclaims
  TradingView affiliation and leaves ToS compliance to the user.
- **Security note from the repo itself.** README warns that copies under other accounts add a
  `package.json` that installs npm malware (MAL-2026-16390). The official project is Python-only.
- **Verdict: REFERENCE-ONLY.** Readable as a map of the scanner endpoint's quirks and failure modes
  (retry/timeout/stale-cache notes in `screener_provider.py`); no file in our repo should change.

## 2. Open-Dev-Society/OpenStock

- **What it does.** A consumer watchlist web app: Next.js 15, MongoDB, Better Auth, Inngest, Gemini for
  welcome emails. Quotes come from Finnhub (`lib/actions/finnhub.actions.ts`). TradingView appears only as
  embeddable iframe widgets (`hooks/useTradingViewWidget.tsx`), not as a data source.
- **License.** AGPL-3.0, LICENSE file present. README states that deploying a modified version "as a web
  service" requires releasing your source under the same license. Copying its code into RoboAlgo (a
  network service on Render) would put RoboAlgo's source under that obligation. This is the hard gate.
- **Maintained.** Yes, last push 2026-10-01.
- **Credentials.** Finnhub key, MongoDB, optional Gemini/Inngest/Gmail. Price alerts are described in code
  and docs as an "OpenStock Cloud" feature, off in the free default mode.
- **Marginal value.** Nothing for our engines. We have no Mongo/Next.js stack; the TradingView widget
  embed is a TradingView product that does not require this repo.
- **Verdict: SKIP.** AGPL plus a consumer-app scope with nothing our backend lacks.

## 3. Mathieu2301/Tradingview-API

- **What it does.** A Node library speaking TradingView's private chart websocket
  (`wss://{data|prodata|widgetdata}.tradingview.com/socket.io/websocket`, `src/client/client.ts`),
  sending a browser `Origin` header and User-Agent. It offers candles, quotes, built-in studies, Pine
  studies/strategies (account cookies required), symbol search, watchlists. Its `getScreener`
  (`src/http/screener.ts`) is the same `scanner.tradingview.com/{market}/scan` POST we already make.
- **License.** No LICENSE file at the repo root (I tried LICENSE, LICENSE.md, LICENSE.txt on main; the
  root listing has none). `package.json` declares `"license": "ISC"`. GitHub reports no license. ISC is
  permissive, so practical risk is low, but we would be relying on a manifest field, not a license
  grant. I would not ship it without the author adding the file.
- **Maintained.** Ambiguous. Around 12 PRs merged 2026-10-03 to 10-05 (V4 rewrite, `4.0.0-rc.1`). The
  default npm tag is still v3 (v3.5.2, 2025-10-05). V4 is "a breaking rewrite ... no compatibility layer"
  and is under the `next` tag. 102 open issues, some from 2022-2023.
- **Runtime cost.** Node >= 20 only. Integration means a Node sidecar service on Render and a bridge to
  Python (HTTP or Redis). The README also steers users to an agent-launch flow (Claude Code / Codex /
  a hosted agent), which is not relevant to us.
- **Credentials.** Without a TradingView account: shorter intraday history, no Pine studies, possibly
  delayed or substitute feeds (README and `docs/data-api.md`). With an account you hand it your
  `sessionid` cookies.
- **What it adds that we lack:** websocket streaming, Pine access, symbol search. We do not currently
  need streaming (5-minute scan, paper/auto-trader). Pine needs a TV account. Symbol search is one HTTP
  call to `symbol-search.tradingview.com`, which I saw in its source and did not test.
- **Indicator hazard: partly, flagged.** `getTechnicalAnalysis` (`src/http/market.ts`) pulls
  `Recommend.All/MA/Other` from `scanner.tradingview.com/global/scan`, i.e. precomputed scanner columns.
  `getScreener` accepts arbitrary column names (its docs example requests `Stoch.RSI.D`). Its chart
  candles default to split adjustment (`adjustment: 'splits'`); I did not test those or the scanner
  ratings on CTVA.
- **ToS.** Highest exposure of the five: an undocumented websocket protocol reimplemented with spoofed
  browser headers, optionally with a user's logged-in session cookies. Its README says only "check your
  data provider's terms".
- **Verdict: SKIP.** Its unique features are ones we do not need, and the price is a Node sidecar,
  a protocol that can change without notice, and no clean license file.

## 4. TauricResearch/TradingAgents

- **What it does.** A LangGraph multi-agent framework of LLM agents (analysts, bull/bear researchers,
  trader, risk debaters, portfolio manager) that outputs a rating. Data comes from Yahoo
  (`yf.Ticker().history(auto_adjust=True)` in `dataflows/vendors/yahoo/ohlcv.py`), SEC EDGAR, FRED,
  Alpha Vantage, Reddit/StockTwits. Indicators are computed with `stockstats` on adjusted OHLCV.
  No TradingView data is used (I did not grep every file; the README, pyproject and the files I read
  show none). That makes it the one repo here that is not exposed to the precomputed-column hazard.
- **License.** Apache-2.0, LICENSE file present. Passes the gate (keep attribution/NOTICE if copying).
- **Maintained.** Very: v0.6.0 on 2026-10-03, near-daily commits.
- **Runtime cost.** Python >= 3.11, so it fits the language. But `pyproject.toml` requires
  `pandas>=3.0.6` plus langchain/langgraph 1.x; our venv has pandas 2.3.3, so adopting it as a dependency
  means a pandas major upgrade across a 70k-LOC codebase. Every analysis also needs LLM API keys and
  paid calls: the paper says 11 LLM calls and 20+ tool calls per prediction.
- **Against our "rule-based personas, no LLM" decision: nothing here changes it.**
  - Its own README says results are non-deterministic across runs and that backtests "are not guaranteed
    to match any published figure ... Treat the framework as a research scaffold ... not as a strategy
    with a fixed, replicable return."
  - Its own `backtest.py` says it "evaluates decision quality. It is not a portfolio simulator": no fills,
    no costs.
  - Its CHANGELOG records look-ahead leaks found in the framework's data path: v0.2.3 (2026-03-29,
    "Backtesting fetchers no longer leak look-ahead data"), v0.3.1, and v0.4.0 (2026-08-31: FRED vintage,
    social sentiment, memory). Any backtest produced before those fixes is suspect. I did not find
    anything saying the paper's runs were redone.
- **Performance claims, attributed (arXiv 2412.20138, HTML v7).** Table 1 reports TradingAgents
  cumulative return 26.62% / 24.36% / 23.21% and Sharpe 8.21 / 6.39 / 5.60 on AAPL / GOOGL / AMZN, against
  buy-and-hold of -5.23% / 7.78% / 17.1%. The authors write that the highest Sharpe "exceeds our expected
  empirical range" and attribute it to "few pullbacks".
  - Out-of-sample? Not in any meaningful sense. The window is Jan 1 to Mar 29, 2024 (three months),
    single period, no held-out period. The text lists five tickers (Apple, Nvidia, Microsoft, Meta,
    Google) while Table 1 shows three (AAPL, GOOGL, AMZN).
  - Multiple-testing corrected? No evidence of it. Keyword search of the paper text: zero hits for
    "overfit", "out-of-sample", "statistical", "slippage", "commission", "fees", "contaminat", "leak".
    ("significan" appears 37 times; I did not read those hits, so I cannot rule out a significance test,
    but I saw none in the results text I read.) The only look-ahead statement is that agents use data "available up to each
    trading day". It does not address whether the LLM backbones (gpt-4o / o1-preview) already knew that
    quarter's prices.
  - These numbers are reported, not corrected. I have no basis to repeat them as facts, and the
    repo cannot reproduce them (per its own README).
- **Idea worth borrowing (minor).** Its `backtest.py`/memory log scores each logged decision by realised
  return and alpha against a regional benchmark over a fixed holding window, grouped by rating, with
  counts. Our `lab_engine/verdict_log.py` docstring says `scorecard()` marks rows against the current
  price. A benchmark-relative, fixed-horizon score would be a more honest measure. Caveat: I read that
  file's docstring and function list, not the `scorecard()` body.
- **Not needed:** its SEC EDGAR "as filed" point-in-time fundamentals. Ours (`lab_engine/fundamentals.py`)
  are live snapshots from yfinance; point-in-time only matters if we backtest personas historically.
- **Verdict: BORROW-IDEA.** File that would change: `lab_engine/verdict_log.py` (scorecard horizon and
  benchmark). Do not add it as a dependency or introduce LLM calls.

## 5. tradesdontlie/tradingview-mcp

- **What it does.** An MCP bridge to the TradingView Desktop app via Chrome DevTools Protocol on port 9222.
  It reads and drives the chart in a locally running, logged-in desktop app: Pine editing, drawings,
  alerts, replay, screenshots. README says "Requires a valid TradingView subscription".
- **License.** MIT text in LICENSE (copyright 2026 tradesdontlie) with appended notices saying the license
  covers the source only and grants no rights to TradingView's software or data. GitHub labels it
  "Other/NOASSERTION" because of the appended text. The grant itself is standard MIT.
- **Maintained.** Weakly. Last commit on main 2026-07-28; a commits query since 2026-07-29 returned
  none. 266 open issues and about 14 recent open PRs from contributors.
- **Runtime cost.** Node 18+, plus TradingView Desktop running with a debug flag on a machine with a GUI.
  It cannot run on Render. It is a single-user desktop tool, not a server component.
- **Its own disclaimer forbids our use case.** Item 4 says it "must not be used for ... Performing
  automated trading or algorithmic decision-making using extracted data"; item 2 admits it "may conflict"
  with TradingView's Terms; item 3 puts account bans on the user.
- **README versus code.** README says it "does not connect to TradingView's servers". Source does:
  `src/core/pine.js` fetches `pine-facade.tradingview.com`, `src/core/alerts.js` posts to
  `pricealerts.tradingview.com`, `src/core/chart.js` calls `symbol-search.tradingview.com`, all with the
  app's authenticated session.
- **Silent wrong data.** Open issue #540 (2026-09-21, reporter-verified on `main`): `data_get_ohlcv` ignores
  `symbol`, `timeframe` and `bars`, returns the active chart's 100 bars, and reports `success: true`.
  I read the issue; I did not reproduce it.
- **Indicator hazard.** It reads values from the desktop chart, not the scanner columns. I did not test
  CTVA through it.
- **Verdict: SKIP.** It might help the user personally with Pine Script work on their own desktop, but it
  does not make RoboAlgo better and its author prohibits algorithmic use.

## What I would actually do

1. Nothing from these five as a dependency. The only capability the TradingView wrappers add that we
   lack is websocket streaming, Pine access and symbol search; we do not need any of them, and each
   comes with higher ToS exposure than our own non-redistributing call.
2. Keep `data_engine/tradingview_provider.py` restricted to `name`, `close`, `volume`. Two of the five
   repos (1 and 3) expose TradingView's precomputed indicator and rating columns as trade signals, which
   is the CTVA hazard.
3. If non-US screening is ever wanted, change the market path in our own provider and test each market
   for the same corporate-action problem first. Do not import a wrapper for it. (Not probed here.)
4. Low priority: in `lab_engine/verdict_log.py`, add a fixed-horizon, benchmark-relative score with counts
   (the TradingAgents idea), after reading the current `scorecard()` to confirm it is not already there.
5. Do not add LLM agents on the strength of TradingAgents' results (see section 4).

## What I did not verify

- No live probe of any repo's code. I did not run CTVA through any of them.
- Non-US market support via the scanner path: taken from source and docs, not tested.
- TradingAgents paper: only the HTML text was searched by keyword; I did not read every section.
  Table 1 numbers are quoted as extracted text.
- GitHub API data was fetched unauthenticated on 2026-10-06; star/issue counts will drift.
- Pricing figures are quoted from atilaahmettaner's README, not checked.
