# Insider purchase clusters and dilution with cash burn on fallen stocks: results, 2026-10-26

**PROVISIONAL: survivors only (delisted names missing).** Pre-registration: `docs/INSIDER_STUDY_PREREG.md`, written 2026-10-25 before any data was downloaded, plus Amendment 1 (the Form 345 URL). Output: `~/.roboalgo/reports/insider_study_results.json`. Not trading advice.

## Verdicts

| Arm | Primary | 97.5% interval | Verdict |
|---|---|---|---|
| Long: insider purchase clusters, 12-month excess return | +1.49% | [-9.63%, +11.89%] | no demonstrated edge |
| Short: dilution plus burn, 12-month wipeout rate difference | +12.67% | [+7.24%, +18.03%] | **PASS** |

Both verdicts are the mechanical application of the pre-registered rule to the pre-registered primary. The long arm's lower bound is below zero; the short arm's is above it.

## Inputs

71 Form 345 quarterly data sets, 2006q1 to 2023q3, 771 MB, giving **218,691 qualifying (owner, filing) purchases** and **68,887 cluster events**. 1,344 companyfacts files for the 1,356 CIKs the universe maps to (12 CIKs have no XBRL facts and read as unknown, as pre-registered). No Tiingo delisted prices, so the price set is survivors only and both results carry the PROVISIONAL stamp.

**24,793 fallen stock-months** across 2,053 symbols. The symbol to CIK map resolves 66.8% of them; the mapped share runs from 46% (2013 to 2015) to 74% (2009, 2023). 2,111 symbols are re-used across two or more CIKs somewhere in the data.

## Long arm: no demonstrated edge

| Group | Rows |
|---|---|
| cluster (treated) | 2,437 |
| control (no qualifying purchase in the window) | 11,768 |
| single buyer (reported separately, never in the control) | 2,331 |
| excluded: unmapped / ambiguous / no quarter coverage | 8,098 / 137 / 22 |

Primary, 12-month excess return over same-month controls: **+1.49%, 97.5% [-9.63%, +11.89%]**, 2,437 rows in 202 month clusters. Stock-clustered robustness: +1.49% [-11.79%, +17.97%].

Secondary rows (95% intervals, descriptive, no verdict attached):

* 6-month excess +0.46% [-3.78%, +4.52%]; 36-month excess +0.55% [-16.12%, +16.31%].
* Single-buyer months, 12-month excess -0.92% [-6.74%, +4.98%].
* **Re-used symbols removed: +1.49% becomes -0.10% [-9.60%, +9.12%]** on 2,095 rows. The headline positive mean does not survive dropping re-used tickers, which is the main reason not to read the +1.49% as a small real effect.
* **12-month wipeout rate, cluster minus control: -3.17% [-5.38%, -0.77%]**. Cluster months halved less often than controls. This is a secondary row and carries no verdict, but it is the only long-arm number whose interval clears zero, and it points at downside rather than upside.

## Short arm: PASS

Classified on 9,459 of 16,558 stock-months with a unique CIK: **668 flagged** (diluted 20% or more over the year AND under one year of cash runway), **8,791 unflagged**, 7,099 unknown (missing or stale XBRL, excluded as pre-registered). First classified month 2010-08; coverage is thin before 2012, as the pre-registration predicted.

**Wipeout rate (price halves within 12 months): flagged 40.42%, unflagged 27.74%, difference +12.67%, 97.5% [+7.24%, +18.03%]** across 158 month clusters. Stock-clustered robustness: +12.67% [+4.65%, +21.36%]. Both intervals clear zero.

### Post-hoc concentration check (not pre-registered, cannot change the verdict)

Classified rows concentrate in recent years (2,101 in 2022, 2,093 in 2023), so the obvious failure mode is a result carried by one bear market. It is not:

| Window | flagged | unflagged | difference | 97.5% interval |
|---|---|---|---|---|
| full, 2010 to 2023 (pre-registered) | 40.4% | 27.7% | +12.7% | [+7.2%, +18.0%] |
| excluding 2022 and 2023 | 35.3% | 25.5% | +9.8% | [+4.0%, +16.7%] |
| excluding 2020, 2022 and 2023 | 49.4% | 32.2% | +17.3% | [+10.5%, +24.7%] |
| 2010 to 2019 only | 46.7% | 28.1% | +18.6% | [+9.8%, +27.7%] |
| 2020 to 2023 only | 38.1% | 27.6% | +10.5% | [+3.7%, +16.5%] |

Year by year the difference is positive in 10 of the 12 years that have both groups. The two negatives are 2012 (8 flagged rows) and **2020 (-4.7%, flagged 5.5% versus unflagged 10.2%)**, the post-crash melt-up in which almost nothing halved. That is the regime to expect this to fail in.

### Which way the survivorship bias runs

For the short arm it runs **against** the finding, as the pre-registration states in advance: the flagged names most likely to die are exactly the ones missing from a survivors-only price set, and a delisted name held flat at its last close never records a wipeout. The true difference is more likely larger than +12.7% than smaller. For the long arm the direction is unknown, because dead cluster names and dead control names are both missing.

## What this does and does not say

* A fallen stock that is diluting heavily and running out of cash halved within a year **40% of the time** against 28% for other fallen stocks, over 2010 to 2023, on surviving names, before costs. That is a usable avoid-or-short filter on this universe.
* It is **not** a demonstrated edge in insider buying. Two or more insiders buying did not raise 12-month returns on this universe, and the one long-arm interval that clears zero says those names fell apart less often, not that they rose more.
* Nothing here is net of costs, borrow or capacity. Fallen stocks between $1 and $10 trading $300,000 a day are expensive to trade and hard to size, and the short arm is the arm that needs borrow.
* The next thing that would change the numbers is the Tiingo delisted load, which lifts the PROVISIONAL stamp and should move the short arm up.

## Reproduce

```bash
export SEC_USER_AGENT="<name> <contact>"
python scripts/insider_study.py --dry-run
python scripts/insider_study.py --workers 4
```

Downloading the inputs: `data_engine.sec_insider.download_quarters([...])` for the Form 345 zips (into `~/.roboalgo/sec/insider`), and `data_engine.sec_edgar.company_facts(cik)` per CIK for the companyfacts cache (`~/.roboalgo/sec/facts`, 3.4 GB for 1,344 companies). Both need `SEC_USER_AGENT`.

`setup_engine/registry/trials.jsonl` was deliberately left at 400 lines by the study build, so these two primary tests are **not** in the trial registry. Whether real-data runs of pre-registered studies should be appended there is an open question, not an oversight of this run.
