# Insider purchase clusters and dilution with cash burn on fallen stocks: pre-registration, 2026-10-25

**Date:** 2026-10-25 · **Status:** pre-registration, written before any data run · **Code:** `scripts/insider_study.py` (CLI), `setup_engine/insider_study.py` (study logic), `data_engine/sec_insider.py` (Form 345 loader) · **Not trading advice**

## Status

This document was written on 2026-10-25, **before any SEC insider data or SEC companyfacts data was downloaded or seen.** None exists locally: there is no `~/.roboalgo/sec/insider`, no `~/.roboalgo/sec/facts` and no `~/.roboalgo/tiingo`. The only things looked at before writing were file listings (the price history `~/.roboalgo/history_2000` holds 58 `batch_*.parquet` files and no `terminal.json`) and the code and results of earlier studies on the same price cache (the multibagger study supplies the "fallen" definition below). No earlier study used SEC insider filings or SEC companyfacts data.

Every parameter below is fixed now. None is tuned after seeing data. If a parameter has to change (for example because a real file format differs from what is assumed here), the change is written at the bottom of this document as a dated amendment with the reason, and the earlier run, if any, is reported next to it.

The code is built and tested offline against hand-made fixtures. Running it on real data is a separate, later step.

## Question

Two questions, asked on the same universe of fallen US stocks (defined below).

1. **Long arm.** When at least two different directors or officers of a fallen stock each buy at least $5,000 of its shares on the open market within 30 days, does the stock earn a higher 12-month return than other fallen stocks that had no insider purchase in the same period?
2. **Short arm.** Among fallen stocks, do the ones that have both diluted their share count by 20% or more over the past year and are burning cash with less than a year of runway fall to half their price or less within 12 months more often than the ones that are not flagged?

Each arm has one primary test, so there are two primary tests. Everything else is descriptive.

## Data and how to obtain it

Nothing in this section has been checked against live files. Every item marked UNVERIFIED is an assumption that the first real download will confirm or break. The loader fails loudly with the columns it found, and the CLI exits with code 3, rather than guessing.

| Input | Where it goes | Status |
|---|---|---|
| Monthly price history (survivors) | `~/.roboalgo/history_2000/batch_*.parquet` | present: 58 batch files; no `terminal.json` |
| SEC Form 3/4/5 quarterly data sets ("Form 345") | zips in `~/.roboalgo/sec/insider/`, one per quarter, named like `2015q1_form345.zip` | UNVERIFIED, missing |
| SEC companyfacts, one JSON per CIK | `~/.roboalgo/sec/facts/CIK##########.json` | UNVERIFIED, missing |
| Delisted prices from Tiingo | `~/.roboalgo/tiingo/out/delisted_*.parquet` | missing |

**Form 345 (UNVERIFIED).** The URL pattern assumed by the helper is `https://www.sec.gov/files/structureddata/data/form-345-data-sets/{year}q{q}_form345.zip`. The study reads three tab-separated tables from each zip, matched by file name without directory or extension and case-insensitively, with headers matched case-insensitively:

* `SUBMISSION`: `ACCESSION_NUMBER`, `FILING_DATE`, `DOCUMENT_TYPE`, `ISSUERCIK`, `ISSUERTRADINGSYMBOL`
* `REPORTINGOWNER`: `ACCESSION_NUMBER`, `RPTOWNERCIK`, `RPTOWNER_RELATIONSHIP`
* `NONDERIV_TRANS`: `ACCESSION_NUMBER`, `TRANS_CODE`, `TRANS_SHARES`, `TRANS_PRICEPERSHARE`, and optionally `TRANS_DATE` (never used for a decision)

Also unverified: the member names, the date format (three are tried: `05-JAN-2015`, `2015-01-05`, `01/05/2015`), how `RPTOWNER_RELATIONSHIP` encodes director and officer, and whether each zip holds the filings made in its quarter. The download helper in `data_engine/sec_insider.py` is not wired to the CLI. If used, it needs the `SEC_USER_AGENT` environment variable set to the user's own contact string (the SEC requires one; there is deliberately no default), it throttles, and it refuses to write anything that is not a zip.

**Companyfacts (UNVERIFIED).** The short arm reads per-company XBRL facts from `~/.roboalgo/sec/facts/`. Two layouts are accepted: the cache written by `data_engine/sec_edgar.py` (`CIK##########.json` holding `{"fetched_at", "data"}`) and the raw per-company JSON from the SEC's bulk companyfacts file. The run does no network access and ignores the cache's age on purpose. A CIK with no file is "unknown", never "unflagged".

**Prices.** Month-end prices come from the same cache as the multibagger study (`scripts/multibagger_study.py` panel builder). When Tiingo delisted files exist, they are added to the panel, with the listed (survivor) rows winning any duplicate stock-month, and their symbols count as terminal.

## Universe and window

* **Origins** are month-ends from **2006-01 to 2023-09** inclusive. 2023-09 is the last origin whose 36-month secondary outcome fits inside the price cache's last complete month (2026-09).
* **Fallen** is the multibagger study's variant A, exactly: month-end close from $1 to $10 inclusive, **and** at least 70% below the trailing 60-month high (`dd60 <= -0.70`; the high includes the current month and needs 12 months of history), **and** mean daily dollar volume of the month of at least $300,000. The three constants are copied from `scripts/multibagger_study.py` and a test asserts they are equal to it. NaN in any input means not fallen.
* A fallen stock-month is one row. The same stock appears once per qualifying month.
* **Origin eligibility and `dd60` come from the raw month grid.** Forward outcomes come from the grid where a delisted name is held flat at its last close, so a held-flat month after a delisting is never an origin.

## Definitions

### Qualifying purchase

A row of a Form 345 filing counts as a purchase only if all of these hold.

* The filing is an original Form 4 or Form 5 (`DOCUMENT_TYPE` is "4" or "5" after trimming and upper-casing). **Amendments (4/A, 5/A) are excluded**, so a late amendment cannot complete a cluster with a stale trade. Form 3 and amendments are still used for symbol spans and quarter coverage (below).
* The row is in the non-derivative table with transaction code `P` (open-market or private purchase), compared after trimming and upper-casing. Awards, sales, exercises and everything else are out.
* The reporting owner is a **director or officer**. The relationship text is split on `,` `;` `|` `/`, each piece is reduced to lowercase letters, and the owner qualifies if any piece is `director` or `officer`. A ten-percent owner with no director or officer role does not qualify.
* The value of the owner's purchases in that filing is **at least $5,000.00**. Value per (owner, filing) is the sum of shares times price over that filing's code-P non-derivative rows. A row with a non-finite or non-positive share count or price adds nothing. The threshold is inclusive (two rows of 600 and 400 shares at $5.00 total exactly $5,000.00 and qualify; 999 shares at $5.00 do not). The transaction rows belong to the accession, so every qualifying owner on a joint filing gets one row carrying the filing's sum.

### Cluster

A **cluster** exists for an issuer (identified by its CIK, never by ticker) when, among its qualifying (owner, filing) rows, the 30 calendar days of **filing dates** up to and including some filing date contain **at least 2 distinct owners and at least 2 distinct accession numbers.** The window is `[filing date - 29 days, filing date]`, date-based, so same-day filings are all included and later dates never are. Two filings 29 days apart fall inside one window; 30 days apart do not.

The second condition exists because one Form 4 can list several reporting owners: **a single joint filing is never a cluster.** The same owner buying twice is not a cluster either.

An **event** is emitted at every filing date where the rule holds (one row per issuer and date). The cluster completion date is the earliest event for the issuer, and any event can make a stock-month treated.

### Event date is the filing date

The event date is the **filing date** of the filing that satisfies the rule. The transaction date is never used. The insider trade can precede its filing by days, and only the filing is public.

### Treatment, control and the origin close

The month-end close is the close of the last trading day. A Form 4 accepted after 4pm that day is public only after that close, so it cannot be traded at that close. For month-end `me` (the calendar month end) let `last_weekday(me)` be `me` if it is a weekday, `me - 1 day` if it is a Saturday and `me - 2 days` if it is a Sunday. Exchange holidays are ignored (a known imprecision, documented here).

Each fallen stock-month falls in exactly one group, tested in this order.

1. **excluded_ambiguous**: the price symbol maps to more than one issuer CIK at the point-in-time date (below).
2. **excluded_unmapped**: the symbol maps to no CIK.
3. **excluded_no_coverage**: some calendar quarter touched by `[me - 89 days, me]` has no filing at all in the loaded insider data (below).
4. **cluster (treated)**: the issuer has a cluster event with `me - 89 days <= event_date < last_weekday(me)`. The window is **90 days** (89 days back plus the month-end), the event lower bound is inclusive (a filing 89 days before `me` counts, 90 days before does not), and an event filed on `last_weekday(me)` itself is not usable.
5. **single**: not treated, but the issuer has some qualifying purchase with `me - 89 days <= filing date <= me` (this includes a cluster event filed on or after `last_weekday(me)`). Reported separately, never in the control.
6. **control**: none of the above. The control pool is stricter than "no cluster": it has **no qualifying purchase at all** filed in the full window `[me - 89 days, me]`.

A single stock can be treated at up to three consecutive month-ends from one event.

### Symbol to CIK map and coverage

Prices are keyed by ticker and Form 345 by issuer CIK, so the join is made point-in-time.

* **Symbol normalisation:** trim, upper-case, replace `.` with `-`. The values `""`, `NONE`, `N/A`, `NA`, `NULL` and `-` count as missing.
* **Symbol spans** come from every SUBMISSION row of every document type that has a symbol, a parsed date and a CIK: for each symbol, issuer and zip, the first and last filing date and the number of filings.
* **Point-in-time date** `pit_date = last_weekday(me) - 1 day`. A symbol maps to a CIK at `pit_date` if some span for that symbol has `first_filed <= pit_date` and `last_filed >= pit_date - 365 days`. Exactly one distinct CIK gives that CIK. Two or more gives "ambiguous". None gives "unmapped". The same map feeds the short arm.
* **Quarter coverage (long arm only):** every calendar quarter (labelled like `2015q2`) touched by `[me - 89 days, me]` must have at least one filing in the loaded data. This guards against a missing zip making treated months look like controls. A second effect is intended: **foreign private issuers never file Form 4**, and without the symbol-map requirement they would sit in the control forever; with it, a ticker that never appears in any filing is unmapped and excluded.
* **Re-used symbols** are symbols linked to two or more distinct CIKs anywhere in the loaded data. They are counted in the output and removed in one secondary row.
* All exclusions are counted by reason in the output.

### Short arm: dilution and burn

Both features are read from SEC companyfacts, **point-in-time by filed date**: a value is visible only if its filing date is on or before the read date, and if a fact was restated the latest value filed on or before that date is used. The read date is `pit_date` (the day before `last_weekday(me)`), so a filing made on the last weekday is never visible.

* **Dilution flag** is True iff `shares_now / shares_then >= 1.20`. It is compared as a ratio, not as `now / then - 1 >= 0.20`, because in floating point `1_200_000 / 1_000_000 - 1` is 0.19999999999999996 and would miss the boundary. Exactly 20% counts as diluted. `shares_now` is the value known at `pit_date`; `shares_then` is the value known at `pit_date - 365 days`. Both counts come from **one concept**: the cover-page `dei` share count (`EntityCommonStockSharesOutstanding`) first, else `us-gaap` `CommonStockSharesOutstanding`, and the first concept with a usable pair wins. `shares_then` must be finite and positive.
* **Burn flag** is True iff operating cash flow is negative and `cash / |operating cash flow| < 1.0` years (cash runway under one year; exactly one year is not burn). Operating cash flow is the latest **annual** value known at `pit_date` (duration of 350 to 380 days, filed on or before `pit_date`). It deliberately does not use the latest duration of any length, which could be a later quarter. Cash is the cash concept plus the first available add-on, as `data_engine/sec_edgar.py` computes it. If operating cash flow is zero or positive the flag is False, whatever the cash.
* **Staleness rule:** a value whose period end is more than **550 days** before `pit_date` counts as missing. `data_engine/sec_edgar.py` documents that it returns the latest value however old; this rule closes that.
* **Classes:** `flagged` iff both flags are known and both True. `unflagged` iff both are known and not both True. **`unknown`** otherwise (either flag missing, no facts file, stale data), and unknown is excluded from both groups. Missing data never lands in the unflagged group.
* The short arm runs over fallen stock-months that have a unique CIK. It does not apply the long arm's quarter-coverage rule.

## Outcomes

All prices are month-end closes on the grid where delisted names are held flat at the last close. With origin close `C[i]` and horizon `h` months, return `r_h = C[i+h] / C[i] - 1`, and an outcome needs a finite `C[i+h]`.

**Primary 1 (long): 12-month excess return.** For each treated stock-month, `r12` minus the mean `r12` of all control stock-months with the same month-end. There is no matching on size, price or volatility. A treated row with no control that month, or without a finite `r12`, drops out and is counted.

**Primary 2 (short): wipeout-rate difference.** A **wipeout** is the minimum of the next 12 month-end closes, `min(C[i+1..i+12])`, being at or below 50% of `C[i]` (equal counts). Gaps are skipped, but the wipeout is undefined unless `C[i+12]` is finite. The outcome is the wipeout rate of `flagged` minus the wipeout rate of `unflagged`.

**Secondary rows (descriptive, 95% intervals, alpha 0.05, no verdict).**

* Long arm: 6-month and 36-month excess return for treated stock-months.
* Long arm: 12-month excess return of `single` stock-months against the same controls.
* Long arm: difference in 12-month wipeout rate, cluster minus control (month clusters).
* Long arm: the 12-month excess return with re-used symbols removed.
* Short arm: the wipeout-rate difference with the bootstrap clustered by stock.

Also reported without a decision attached: all group counts, the control summary, the mapped share by year, the number of re-used symbols, the insider quarters and per-file row counts, the number of classified short-arm rows by year and the first classified month.

## Statistics

* **Primary intervals are month-clustered bootstrap intervals**: whole month-ends are resampled with replacement, **2000 draws, seed 0**, using `setup_engine/trade_stats.py`. The mean is a ratio of sums (each stock-month weighs the same; a month with more rows weighs more). The long arm resamples the excess returns by month-end; the short arm resamples the two groups by month-end.
* **`alpha = 0.025`** is passed to the bootstrap, which takes the 1.25% and 98.75% quantiles. That is a **two-sided 97.5% interval**: a Bonferroni split of 5% over the two primary tests. The "lower bound" in the verdict rule is the lower end of this interval.
* **Stock-clustered intervals are reported as robustness** (the same quantity resampled by ticker, at 97.5% for both primary outcomes). They do not enter the verdict. They matter because the same stock appears in overlapping months and in overlapping 12-month windows, which month clustering does not absorb; `trade_stats.py` itself notes intervals are still somewhat too narrow.
* No other test is a decision test. Several secondary rows are printed, and a secondary interval just above zero is not evidence.

## Verdict rule

Mechanical, applied to the two primaries only:

**LONG PASS iff the 97.5% lower bound of the 12-month excess return is > 0; SHORT PASS iff the 97.5% lower bound of (flagged wipeout rate minus unflagged wipeout rate) is > 0; otherwise no demonstrated edge.**

Details that follow from it:

* The test is strict: **a lower bound of exactly 0 is not a PASS.**
* If the interval cannot be computed (fewer than two month clusters), the verdict is "no demonstrated edge (interval not computable)".
* If the companyfacts directory is missing, the short verdict is "not run: no companyfacts directory", not a failure and not a pass.
* The wording "no demonstrated edge" is the whole negative result. It is not "the effect is zero".
* Robustness and secondary rows cannot change a verdict. If the stock-clustered interval disagrees with the month-clustered one, both are reported and the verdict still follows the primary.

## PROVISIONAL rule

The result is **PROVISIONAL** iff no delisted names are in the price set: no delisted symbol is loaded from the Tiingo directory's `delisted_*.parquet` files **or** listed in the history directory's `terminal.json`. The run decides this from the symbols actually loaded, so an empty `terminal.json` or Tiingo files holding no symbols do not lift it. Today both are absent, so the first real run will be PROVISIONAL.

When PROVISIONAL, the output JSON sets `provisional` to true, the first caveat is the text below, and every verdict line starts with it:

`PROVISIONAL: survivors only (delisted names missing)`

for example `PROVISIONAL: survivors only (delisted names missing) | PASS`. The rule and the thresholds are the same either way; the stamp tells the reader what the number is. The price cache holds stocks listed today, so the stocks that were fallen and died are missing. Direction: for the long arm it is unknown (dead insider-buying names and dead control names are both missing). For the short arm it is the unfavourable case: the flagged names most likely to die are the ones missing, which pushes the flagged wipeout rate down.

## Known biases and risks

* **Survivorship** (above). Holding a delisted name flat at its last close also understates wipeouts for bankruptcies, the same convention the multibagger study states.
* **Unverified inputs.** The Form 345 format and the companyfacts layout have not been seen. Possible surprises: the relationship split into flag columns, other date formats, CRLF, embedded tabs (such rows are skipped and counted as rows read per table).
* **Ticker join.** It depends on the as-reported symbol matching the price-cache symbol after normalisation. The run reports the mapped share by year, so a low hit rate shows. Re-used tickers are counted, ambiguous months are excluded, and one secondary row repeats the long-arm test with re-used symbols removed. Symbol spans are as reported at the time, which keeps the map point-in-time but cannot repair a company that changed ticker mid-window.
* **XBRL coverage.** Smaller reporting companies began XBRL tagging for periods after about mid-2011, and the dilution test needs a share count a year earlier, so the short arm's effective window is expected to begin around 2012 although origins begin in 2006. Before that most rows are `unknown`, not unflagged. The output reports classified rows by year and the first classified month.
* **Share counts are as reported, not split-adjusted.** A forward split between `shares_then` and `shares_now` can raise a false dilution flag. The cover-page count is dated near the filing, not the balance-sheet date, and a multi-class company can report one class only (`data_engine/sec_edgar.py` documents both). Cash is cash plus one add-on, and debt, converts and leases are not in the burn test.
* **Price artifacts.** Yahoo adjusts history retroactively for splits, which can make serial reverse-splitters look fallen. The fallen definition inherits this from the multibagger study.
* **Overlap and serial dependence.** One event treats up to three consecutive month-ends, and 12-month windows overlap. Month clustering handles same-month dependence, not same-stock dependence (hence the stock-clustered robustness rows).
* **Look-ahead traps closed by design:** trade date versus filing date; a filing accepted after the last close; restatements (latest value filed on or before the read date); stale XBRL values; a held-flat month used as an origin; a symbol map built from later filings.
* **Calendar.** Exchange holidays are ignored in `last_weekday`. The filing time of day is not available, so an event filed on the last weekday is excluded even if it was filed before the close.
* **Only open-market purchases by directors and officers.** Purchases by ten-percent owners, grants and option exercises are out by design, and what an officer or director is depends on the unverified relationship field.
* **Costs and capacity are not modelled.** Fallen stocks under $10 trading $300,000 a day are costly and hard to size.

## What is not claimed

* No result is claimed. This document contains no numbers from data.
* A PASS would mean the stated interval cleared zero on this sample and this universe. It would not mean the strategy is tradable after costs, nor that it works outside 2006-2023, nor that it works on stocks that were not fallen.
* A PROVISIONAL result is a survivors-only result and is labelled as such.
* The grid of definitions here (window, thresholds, roles) is one fixed choice. Nothing here shows it is the best one, and no neighbouring settings will be tried to rescue a negative result.
* `setup_engine/registry/trials.jsonl` is deliberately not appended by this build. It stays at 400 lines. No real-data run has happened, so there are no trials to record.

## Reproduce

```bash
python scripts/insider_study.py --dry-run     # reports which inputs are present; computes nothing; exit 0
python scripts/insider_study.py               # the study; writes ~/.roboalgo/reports/insider_study_results.json
python -m pytest tests/test_data_engine/test_sec_insider.py tests/test_study/test_insider_study.py tests/test_study/test_insider_study_cli.py -q
```

Options: `--history`, `--insider-dir`, `--facts-dir`, `--tiingo-dir`, `--out`, `--workers`, `--dry-run`. Exit codes: 0 for a normal run or a dry run, 2 if a required input is missing (the price history or the insider zips), 3 if the insider data does not match the assumed format (the message lists the columns found). On any of these the CLI prints a message, not a traceback. The output is strict JSON (no NaN or Infinity).

## Files

* `docs/INSIDER_STUDY_PREREG.md`: this document.
* `scripts/insider_study.py`: the CLI; its docstring repeats these definitions.
* `setup_engine/insider_study.py`: groups, outcomes, short-arm features and verdicts (pure functions).
* `data_engine/sec_insider.py`: the Form 345 loader, cluster builder, symbol map and the unwired download helper.
* `tests/test_data_engine/test_sec_insider.py`, `tests/test_study/test_insider_study.py`, `tests/test_study/test_insider_study_cli.py`: fixture tests, including hand-computed boundaries and an end-to-end run on synthetic data.
