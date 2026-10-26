# Do unadjusted splits and spin-offs change the earlier studies? No

**Date:** 2026-10-12 · **Code:** `setup_engine/artifacts.py`, `scripts/artifact_sensitivity.py` · **Raw output:** `docs/artifact_sensitivity_results.json`

**Rule (post-hoc, not pre-registered):** a day is a suspected artifact when the close moves 50% or more in a day, the prior day traded at least $1M, and the day's volume is under twice the prior 20-day average (a split or spin-off has no volume spike; real news does). A trade or event row is "contaminated" when such a day falls in its window (turnaround trades: 60 days before the signal to the exit; event study: the event week to 156 weeks later). The control is left uncleaned on purpose.

**Found:** 223 flagged days in 168 of 3,982 stocks scanned.

| Study | Before | After |
|---|---|---|
| Turnaround rule, primary (resolved trades) | n 1,568, mean +0.0957R, month interval [-0.112, +0.320] | n 1,562, mean +0.0902R, [-0.118, +0.314] |
| Event study S1 excess | n 1,539, -0.0076 [-0.070, +0.061] | n 1,538, -0.0075 [-0.070, +0.061] |
| S2 | n 251, -0.0403 | n 250, -0.0401 |
| S3, S4 | unchanged (no flagged rows) | |
| B0 | n 6,672, +0.0038 | n 6,660, +0.0041 |

The 6 dropped turnaround trades were mostly big winners (mean +1.52R: SDEV +5.8R, OCGN +2.8R, GTEC +2.3R), so removing them lowers the result. Only 0.3% of the control rows are contaminated. **No conclusion changes.** The "before" numbers reproduce the published ones exactly, including intervals and seeds.

**Limits.** Yahoo back-adjusts old splits, so a clean result is not proof the data is clean. The rule also flags halt-reopen moves and low-volume buyout gap-ups, which are real outcomes, and misses artifacts that arrive with volume.
