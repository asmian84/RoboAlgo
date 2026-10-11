# Scanner recall: would our scanners have flagged the big winners near their lows?

Data through 2026-10-06. Weekly rows from 2021-06-01. This is a DESCRIPTION of existing scanners against hindsight winners: the 46 names were chosen because they went up (selection on the outcome), the low of each is found ex post (`setup_engine.mtf.best_run` on weekly closes since 2021-10-08), and the price cache holds today's listings only (survivorship). A high recall is not evidence of an edge: a scanner that flags everything has perfect recall.

## What each scanner looks for

- **Finder presets** (`finder:signal`, `finder:pennyking`, `finder:maker`): A Finder flag on day t means ALL of: a golden cross (50-day average above the 200-day average and the close above the 50-day average); a compressed range among the last 30 compression rows with a compression score >= 50; no open gap above the preset's max_gap_pct in the last 20 bars; a 20-day average volume >= the preset's min_volume; the close inside the preset's price band; at least finder_backtest.MIN_ROWS bars of history; and valid trade levels (entry, stop, target) taken from the compressed range's driver bar. Presets: signal (no price band, 100k volume, 3% gaps), pennyking (price <= 5, 500k volume, 8% gaps), maker (price 5-20, 250k volume, 6% gaps).
- **turnaround**: the pre-registered rule of `scripts/study_turnaround.py` as raw daily signals (70% or more below the 2-year high with a higher low on the prior bar, then a new 20-day high on more than twice the usual volume; price over $1, 20-day dollar volume over 1M).
- **registry**: the 20 registered chart setups that have a rule, read on Friday observations with cross-sectional ranks against every stock with a 50-day dollar volume of 5e5 or more. The 3 approved round-2 textbook rules (provisional) have their own columns; `learned` counts how many of the 17 round-1 learned rules (all rejected on robustness) flagged the name in the window. The 3 round-1 textbook entries have empty rules and cannot be evaluated.
- **mtf:S1..S4, mtf:B0**: the five pre-registered multi-timeframe signatures of `setup_engine/mtf.py`:
  - `S1` RS leader: Performance Rank (13 Weeks) = Top 20%; Weekly SMA30 = Price above; Weekly SMA30 Slope = Rising
  - `S2` beaten-down reclaim: Weekly High (52w) = 30% or more below; Weekly SMA30 = Price above; Performance Rank (26 Weeks) = Top 20%
  - `S3` breakout leader: Weekly High (26w) = New High; Weekly Volume = Over 1.5x 10-week average; Weekly SMA30 = Price above; Weekly SMA30 Slope = Rising; Monthly SMA10 = Price above
  - `S4` full confluence: Weekly High (26w) = New High; Weekly Volume = Over 1.5x 10-week average; Weekly SMA30 = Price above; Weekly SMA30 Slope = Rising; Monthly SMA10 = Price above; RS Line = New 26-week high; Benchmark Weekly SMA30 = Price above
  - `B0` market beta baseline: Weekly SMA30 = Price above

## How to read a cell

`+Nw (+P%) / K`: the first week flagged in [low - 8 weeks, low + 52 weeks] was N weeks after the low week (negative = before); the weekly close that week was P% above the low's close; K is how many OTHER liquid stocks the same scanner flagged that same week. `-` means no flag in the window. K is a precision proxy: a scanner that flags hundreds of stocks a week is not singling the name out. Liquid means `mtf.is_liquid`: weekly close of 2 or more, median 13-week dollar volume of 5e6 or more, at least 60 weekly bars. `*` after a name marks a run that spans a gap of more than 30 days between weekly bars (halted or stitched data: do not interpret).

Universe: 10980 stocks scanned, 780580 weekly rows kept, 778713 liquid stock-weeks over 279 weeks (median 2692 liquid stocks a week). Cohort names that are not in the stock cache, or that sit outside a limited run's batches, are ranked against the universe but are not part of it.

## Results

### Existing scanners

| name | low week | run | finder:signal | finder:pennyking | finder:maker | turnaround | r2:compr_gc | r2:stage2 | r2:tight_base | learned (of 17) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMD | 2022-10-14 | 11.3x | +23w (+75%) / 282 | - | - | - | -8w (+72%) / 1146 | +32w (+127%) / 446 | - | 2/17 |
| NVDA | 2022-10-14 | 20.8x | +15w (+81%) / 752 | - | +15w (+81%) / 143 | - | +6w (+45%) / 927 | +25w (+141%) / 422 | - | 2/17 |
| MSFT | 2022-11-04 | 2.4x | +20w (+27%) / 282 | - | - | - | +4w (+15%) / 1201 | +28w (+44%) / 458 | +28w (+44%) / 137 | 0/17 |
| AMZN | 2022-12-30 | 3.3x | +21w (+43%) / 403 | - | - | - | +7w (+16%) / 922 | +41w (+55%) / 370 | +44w (+65%) / 196 | 0/17 |
| TSLA | 2023-01-06 | 4.3x | +40w (+122%) / 421 | - | - | - | +6w (+84%) / 922 | +45w (+107%) / 408 | - | 1/17 |
| AVGO | 2022-10-14 | 10.5x | +13w (+36%) / 689 | - | - | - | +6w (+24%) / 927 | +23w (+49%) / 389 | +30w (+48%) / 139 | 0/17 |
| MU | 2022-12-30 | 22.7x | +13w (+21%) / 440 | - | - | - | +5w (+25%) / 721 | +24w (+35%) / 528 | +30w (+42%) / 247 | 0/17 |
| QCOM | 2023-05-12 | 2.4x | +30w (+28%) / 453 | - | - | - | +27w (+25%) / 162 | +36w (+47%) / 634 | +35w (+35%) / 273 | 0/17 |
| INTC | 2024-09-06 | 7.1x | - | - | - | - | +11w (+30%) / 351 | - | - | 0/17 |
| TXN | 2023-10-27 | 2.3x | +17w (+14%) / 520 | - | - | - | +9w (+19%) / 1299 | - | +30w (+39%) / 362 | 0/17 |
| AMAT | 2022-10-14 | 8.4x | +15w (+50%) / 752 | - | - | - | +7w (+43%) / 1201 | +31w (+70%) / 458 | +33w (+80%) / 178 | 0/17 |
| LRCX | 2022-10-14 | 12.4x | +15w (+53%) / 752 | - | - | - | +7w (+45%) / 1201 | +29w (+68%) / 442 | - | 2/17 |
| KLAC | 2022-10-14 | 9.8x | +13w (+59%) / 689 | - | - | - | -7w (+36%) / 1047 | +16w (+55%) / 411 | +31w (+61%) / 137 | 0/17 |
| MRVL | 2023-01-06 | 8.6x | +29w (+80%) / 910 | - | - | - | +6w (+22%) / 922 | +29w (+80%) / 630 | - | 0/17 |
| ON | 2025-04-04 | 3.6x | +47w (+97%) / 671 | - | - | - | +8w (+25%) / 1082 | +45w (+114%) / 737 | - | 2/17 |
| NXPI | 2022-10-14 | 2.3x | +15w (+31%) / 752 | - | - | - | -8w (+31%) / 1146 | +35w (+43%) / 528 | +46w (+52%) / 288 | 0/17 |
| MCHP | 2025-04-04 | 2.8x | +14w (+106%) / 678 | - | - | - | +8w (+60%) / 1082 | +42w (+106%) / 754 | - | 0/17 |
| ADI | 2022-10-14 | 3.2x | +13w (+24%) / 689 | - | - | - | +8w (+23%) / 568 | +16w (+31%) / 411 | +16w (+31%) / 160 | 0/17 |
| MPWR | 2022-10-14 | 5.3x | +23w (+57%) / 282 | - | - | - | -8w (+65%) / 1146 | +32w (+68%) / 446 | - | 0/17 |
| TER | 2025-04-04 | 6.5x | +22w (+75%) / 789 | - | - | - | +12w (+31%) / 756 | +32w (+147%) / 607 | - | 0/17 |
| SMCI | 2021-10-22 | 32.3x | -8w (+1%) / 812 | -8w (+1%) / 6 | - | - | -5w (+6%) / 267 | +6w (+22%) / 518 | -7w (+8%) / 488 | 0/17 |
| TSM | 2022-10-28 | 7.6x | +21w (+50%) / 283 | - | - | - | +6w (+30%) / 569 | +32w (+66%) / 502 | - | 0/17 |
| ASML | 2022-10-14 | 5.1x | +17w (+73%) / 795 | - | - | - | -8w (+44%) / 1147 | +23w (+71%) / 390 | +34w (+89%) / 193 | 0/17 |
| ARM | 2023-10-20 | 9.2x | - | - | - | - | +10w (+57%) / 1300 | - | - | 0/17 |
| ALAB | 2024-09-06 | 10.4x | - | - | - | - | +3w (+34%) / 440 | +46w (+206%) / 567 | - | 0/17 |
| CRDO | 2023-05-05 | 35.6x | +8w (+127%) / 731 | - | +8w (+127%) / 146 | - | +10w (+121%) / 882 | +13w (+116%) / 614 | - | 0/17 |
| COHR | 2023-05-12 | 14.0x | +14w (+24%) / 452 | - | - | - | +11w (+67%) / 750 | +39w (+124%) / 690 | - | 0/17 |
| LITE | 2023-10-27 | 30.1x | +13w (+53%) / 1015 | - | - | - | +9w (+45%) / 1299 | +47w (+70%) / 690 | - | 0/17 |
| ANET | 2022-06-17 | 9.1x | +16w (+28%) / 167 | - | - | - | +8w (+39%) / 746 | +36w (+49%) / 464 | - | 0/17 |
| VRT | 2022-07-01 | 44.7x | +28w (+81%) / 689 | - | +23w (+65%) / 75 | - | +8w (+43%) / 1047 | +39w (+73%) / 481 | - | 2/17 |
| WDC | 2022-12-23 | 32.3x | +27w (+24%) / 731 | - | - | - | -3w (+16%) / 1201 | +39w (+48%) / 465 | +39w (+48%) / 190 | 0/17 |
| STX | 2022-12-23 | 21.2x | +14w (+31%) / 440 | - | - | - | +4w (+17%) / 399 | +40w (+31%) / 462 | +45w (+43%) / 196 | 2/17 |
| DELL | 2022-09-30 | 16.6x | +32w (+31%) / 394 | - | - | - | +10w (+22%) / 568 | +38w (+49%) / 520 | +38w (+49%) / 187 | 0/17 |
| GLW | 2023-10-27 | 8.3x | +18w (+22%) / 539 | - | - | - | +9w (+15%) / 1299 | +37w (+72%) / 643 | +31w (+40%) / 371 | 0/17 |
| CIEN | 2022-10-14 | 14.7x | +15w (+29%) / 752 | - | - | - | -7w (+35%) / 1047 | - | - | 0/17 |
| CLS | 2022-09-30 | 49.8x | +12w (+29%) / 330 | - | +12w (+29%) / 77 | - | -6w (+34%) / 1146 | +19w (+59%) / 428 | - | 0/17 |
| SNDK | 2025-04-04 | 72.6x | - | - | - | - | +9w (+30%) / 1283 | +52w (+2230%) / 633 | - | 0/17 |
| NBIS | 2024-11-15 | 16.3x | - | - | - | - | +29w (+174%) / 1283 | - | - | 0/17 |
| PLTR | 2022-12-23 | 31.9x | +30w (+161%) / 957 | - | +16w (+40%) / 101 | - | +13w (+30%) / 67 | +25w (+159%) / 528 | - | 0/17 |
| APP | 2023-01-06 | 74.2x | +23w (+147%) / 660 | - | - | +31w (+300%) / 3 | +8w (+44%) / 757 | +34w (+344%) / 653 | - | 11/17 |
| META | 2022-11-04 | 8.6x | +17w (+104%) / 644 | - | - | - | +8w (+33%) / 524 | +27w (+158%) / 425 | - | 10/17 |
| GOOGL | 2022-11-04 | 4.6x | +31w (+41%) / 613 | - | - | - | +17w (+8%) / 757 | +40w (+50%) / 596 | +34w (+38%) / 207 | 0/17 |
| AAPL | 2023-01-06 | 2.6x | +11w (+24%) / 282 | - | - | - | +6w (+18%) / 922 | +18w (+33%) / 425 | +18w (+33%) / 139 | 0/17 |
| SPY | 2022-09-30 | 2.2x | +18w (+15%) / 831 | - | - | - | -7w (+20%) / 747 | - | +35w (+20%) / 179 | 0/17 |
| QQQ | 2022-10-14 | 2.9x | +22w (+17%) / 272 | - | - | - | -8w (+24%) / 1147 | +32w (+34%) / 447 | +32w (+34%) / 125 | 0/17 |
| SMH | 2022-10-14 | 7.6x | +15w (+38%) / 753 | - | - | - | +6w (+28%) / 928 | +31w (+54%) / 459 | +30w (+42%) / 140 | 0/17 |

### Multi-timeframe signatures (pre-registered)

| name | low week | run | mtf:S1 | mtf:S2 | mtf:S3 | mtf:S4 | mtf:B0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMD | 2022-10-14 | 11.3x | +24w (+75%) / 349 | - | - | - | +15w (+35%) / 1734 |
| NVDA | 2022-10-14 | 20.8x | +15w (+81%) / 393 | - | +32w (+247%) / 30 | +32w (+247%) / 25 | +4w (+45%) / 1490 |
| MSFT | 2022-11-04 | 2.4x | +22w (+32%) / 362 | - | +25w (+39%) / 36 | +25w (+39%) / 24 | +13w (+17%) / 1950 |
| AMZN | 2022-12-30 | 3.3x | +21w (+43%) / 437 | - | - | - | +13w (+23%) / 1332 |
| TSLA | 2023-01-06 | 4.3x | +23w (+130%) / 418 | - | - | - | +12w (+83%) / 1332 |
| AVGO | 2022-10-14 | 10.5x | +12w (+38%) / 351 | - | +32w (+90%) / 30 | +32w (+90%) / 25 | +4w (+21%) / 1490 |
| MU | 2022-12-30 | 22.7x | +15w (+25%) / 422 | - | +51w (+73%) / 60 | +51w (+73%) / 37 | -7w (+25%) / 1490 |
| QCOM | 2023-05-12 | 2.4x | +28w (+23%) / 404 | - | +31w (+38%) / 346 | +31w (+38%) / 220 | -7w (+20%) / 989 |
| INTC | 2024-09-06 | 7.1x | +28w (+28%) / 430 | +27w (+27%) / 69 | - | - | +23w (+25%) / 1528 |
| TXN | 2023-10-27 | 2.3x | +29w (+36%) / 493 | - | +26w (+24%) / 40 | - | +7w (+18%) / 1762 |
| AMAT | 2022-10-14 | 8.4x | +14w (+47%) / 369 | - | +22w (+64%) / 30 | - | +4w (+48%) / 1490 |
| LRCX | 2022-10-14 | 12.4x | +23w (+59%) / 327 | - | +16w (+68%) / 189 | +16w (+68%) / 128 | +4w (+59%) / 1490 |
| KLAC | 2022-10-14 | 9.8x | +9w (+46%) / 420 | - | - | - | -8w (+41%) / 1327 |
| MRVL | 2023-01-06 | 8.6x | +20w (+81%) / 437 | - | - | - | +4w (+29%) / 1950 |
| ON | 2025-04-04 | 3.6x | +39w (+68%) / 586 | - | +45w (+114%) / 119 | +45w (+114%) / 115 | +10w (+51%) / 1195 |
| NXPI | 2022-10-14 | 2.3x | +38w (+48%) / 442 | - | +16w (+38%) / 189 | +16w (+38%) / 128 | -8w (+31%) / 1327 |
| MCHP | 2025-04-04 | 2.8x | +13w (+102%) / 386 | - | +44w (+110%) / 289 | +44w (+110%) / 258 | +6w (+69%) / 1268 |
| ADI | 2022-10-14 | 3.2x | +18w (+41%) / 397 | - | +18w (+41%) / 77 | +18w (+41%) / 64 | -8w (+23%) / 1327 |
| MPWR | 2022-10-14 | 5.3x | -8w (+65%) / 275 | - | +32w (+68%) / 30 | - | -8w (+65%) / 1327 |
| TER | 2025-04-04 | 6.5x | +24w (+74%) / 519 | - | +30w (+164%) / 95 | +30w (+164%) / 69 | +17w (+52%) / 1488 |
| SMCI | 2021-10-22 | 32.3x | +3w (+24%) / 441 | - | +28w (+45%) / 18 | +42w (+85%) / 30 | -7w (+8%) / 1401 |
| TSM | 2022-10-28 | 7.6x | +13w (+50%) / 394 | - | +30w (+66%) / 31 | +30w (+66%) / 26 | +3w (+33%) / 1406 |
| ASML | 2022-10-14 | 5.1x | +7w (+60%) / 380 | - | +15w (+76%) / 40 | +32w (+94%) / 26 | +4w (+52%) / 1491 |
| ARM | 2023-10-20 | 9.2x | - | - | - | - | +24w (+161%) / 1865 |
| ALAB | 2024-09-06 | 10.4x | +37w (+137%) / 423 | - | - | - | +5w (+67%) / 1754 |
| CRDO | 2023-05-05 | 35.6x | +5w (+116%) / 418 | - | +30w (+150%) / 77 | +30w (+150%) / 46 | +3w (+80%) / 1010 |
| COHR | 2023-05-12 | 14.0x | +5w (+97%) / 418 | - | +39w (+124%) / 81 | +39w (+124%) / 46 | +3w (+40%) / 1179 |
| LITE | 2023-10-27 | 30.1x | +36w (+53%) / 480 | - | - | - | -8w (+49%) / 1480 |
| ANET | 2022-06-17 | 9.1x | +10w (+35%) / 314 | - | +35w (+52%) / 77 | - | +6w (+28%) / 1040 |
| VRT | 2022-07-01 | 44.7x | +19w (+85%) / 332 | +10w (+61%) / 80 | +47w (+138%) / 30 | +47w (+138%) / 25 | +6w (+63%) / 1508 |
| WDC | 2022-12-23 | 32.3x | +32w (+42%) / 412 | - | +49w (+58%) / 77 | - | +5w (+47%) / 1734 |
| STX | 2022-12-23 | 21.2x | +36w (+46%) / 386 | - | - | - | +5w (+37%) / 1734 |
| DELL | 2022-09-30 | 16.6x | +30w (+27%) / 467 | - | +38w (+49%) / 32 | +38w (+49%) / 19 | +8w (+31%) / 1499 |
| GLW | 2023-10-27 | 8.3x | +31w (+40%) / 495 | - | +27w (+27%) / 95 | +27w (+27%) / 69 | +13w (+16%) / 1784 |
| CIEN | 2022-10-14 | 14.7x | +12w (+30%) / 351 | - | - | - | -8w (+41%) / 1327 |
| CLS | 2022-09-30 | 49.8x | +11w (+31%) / 420 | - | +17w (+57%) / 39 | +18w (+63%) / 128 | -7w (+35%) / 1508 |
| SNDK | 2025-04-04 | 72.6x | +52w (+2230%) / 577 | - | +40w (+1153%) / 112 | +40w (+1153%) / 78 | +22w (+128%) / 2101 |
| NBIS | 2024-11-15 | 16.3x | - | - | +43w (+414%) / 83 | +43w (+414%) / 66 | +26w (+112%) / 1269 |
| PLTR | 2022-12-23 | 31.9x | +15w (+29%) / 362 | - | +23w (+131%) / 40 | +23w (+131%) / 19 | +6w (+34%) / 1950 |
| APP | 2023-01-06 | 74.2x | +19w (+158%) / 471 | +18w (+121%) / 30 | +31w (+300%) / 51 | +31w (+300%) / 40 | +12w (+62%) / 1332 |
| META | 2022-11-04 | 8.6x | +15w (+90%) / 397 | - | +25w (+165%) / 36 | +25w (+165%) / 24 | +11w (+54%) / 1591 |
| GOOGL | 2022-11-04 | 4.6x | +25w (+24%) / 467 | - | - | - | +13w (+21%) / 1950 |
| AAPL | 2023-01-06 | 2.6x | +14w (+27%) / 422 | - | - | - | -7w (+17%) / 1405 |
| SPY | 2022-09-30 | 2.2x | - | - | - | - | -7w (+20%) / 1509 |
| QQQ | 2022-10-14 | 2.9x | +25w (+22%) / 363 | - | - | - | -8w (+24%) / 1328 |
| SMH | 2022-10-14 | 7.6x | +15w (+38%) / 394 | - | +16w (+44%) / 190 | +16w (+44%) / 129 | +4w (+28%) / 1491 |

## Recall next to the flag rate

| scanner | flagged in window | ever flagged | flag rate (liquid stock-weeks) | median flagged / week |
| --- | --- | --- | --- | --- |
| finder_signal | 41/46 | 43/46 | 18.59% | 524 |
| finder_pennyking | 1/46 | 1/46 | 0.74% | 18 |
| finder_maker | 5/46 | 6/46 | 4.36% | 117 |
| turnaround | 1/46 | 2/46 | 0.11% | 2 |
| registry:r1-01 | 1/46 | 1/46 | 0.56% | 14 |
| registry:r1-02 | 1/46 | 1/46 | 1.08% | 25 |
| registry:r1-03 | 7/46 | 12/46 | 0.70% | 10 |
| registry:r1-04 | 1/46 | 1/46 | 1.43% | 36 |
| registry:r1-05 | 1/46 | 2/46 | 1.21% | 29 |
| registry:r1-06 | 1/46 | 2/46 | 1.22% | 30 |
| registry:r1-07 | 1/46 | 2/46 | 1.22% | 30 |
| registry:r1-08 | 1/46 | 2/46 | 1.23% | 30 |
| registry:r1-09 | 0/46 | 0/46 | 2.28% | 65 |
| registry:r1-10 | 2/46 | 3/46 | 3.87% | 110 |
| registry:r1-11 | 8/46 | 12/46 | 1.19% | 21 |
| registry:r1-12 | 1/46 | 2/46 | 1.22% | 24 |
| registry:r1-13 | 2/46 | 4/46 | 4.25% | 99 |
| registry:r1-14 | 2/46 | 4/46 | 4.26% | 99 |
| registry:r1-15 | 1/46 | 1/46 | 1.06% | 20 |
| registry:r1-16 | 2/46 | 4/46 | 3.03% | 71 |
| registry:r1-17 | 2/46 | 4/46 | 3.04% | 71 |
| registry:r2-textbook_compression_golden_cross | 46/46 | 46/46 | 11.90% | 271 |
| registry:r2-textbook_stage2_trend | 40/46 | 46/46 | 18.29% | 553 |
| registry:r2-textbook_tight_base_breakout | 21/46 | 33/46 | 7.77% | 202 |
| mtf:S1 | 43/46 | 45/46 | 15.93% | 443 |
| mtf:S2 | 3/46 | 15/46 | 1.40% | 37 |
| mtf:S3 | 31/46 | 44/46 | 1.80% | 37 |
| mtf:S4 | 26/46 | 44/46 | 1.17% | 23 |
| mtf:B0 | 46/46 | 46/46 | 50.70% | 1489 |

## Caveats

- Ex-post selection: the names and their lows are chosen with hindsight. Nothing here validates a scanner.
- Survivorship: today's listings only; delisted losers that every scanner may have flagged are absent, so the flag rates and K are lower bounds on how noisy a scanner was in real time.
- Yahoo adjusts old prices retroactively for splits and the data has no dividend adjustment.
- The registered rules were learned on weekly observations of 2000-2024 data; this table reads them on 2021-2026 Fridays against the full cache's rank universe.
- Weeks are complete weeks only: flags on the final, part-formed week are not attributed to any row.
