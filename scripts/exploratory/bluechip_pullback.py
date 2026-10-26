import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from setup_engine import learn as L
C = L.Condition
UP = (C("W_ma30_50", ">", 0.0), C("W_ma50_slope", ">", 0.0), C("W_c_ma30", ">", 0.0))
RULES = {   # three variants, fixed before any result was looked at
  "P1 uptrend + 5-15% below the 50-day high":            L.Rule(UP + (C("D_dist_hi50", ">=", -0.15), C("D_dist_hi50", "<=", -0.05))),
  "P2 uptrend + within 3% of the 50-day average":         L.Rule(UP + (C("D_c_ma50", ">=", -0.03), C("D_c_ma50", "<=", 0.03))),
  "P3 uptrend + 10-20% below the 50-day high":            L.Rule(UP + (C("D_dist_hi50", ">=", -0.20), C("D_dist_hi50", "<=", -0.10))),
}
need = {"symbol", "date", "dvol50", "fwd_ret_6m"}
for r in RULES.values(): need |= {c.feature for c in r.conditions}
obs = pd.read_parquet("/Users/asmian/.roboalgo/setups/obs.parquet", columns=sorted(need)); obs = obs[obs.fwd_ret_6m.notna()].copy(); obs["ret"] = obs.fwd_ret_6m
obs["rank_dv"] = obs.groupby("date").dvol50.rank(ascending=False, method="first"); bc = obs[obs.rank_dv <= 500].reset_index(drop=True)
bc["era"] = pd.cut(bc.date.dt.year, [2000, 2009, 2017, 2026], labels=["2001-09", "2010-17", "2018-26"])
print("top-500 blue chips at the time; 6-month forward return of flagged minus the same-date blue-chip average (95% CI)\n")
for rn, rule in RULES.items():
    cells = []
    for era, g in bc.groupby("era", observed=True):
        g = g.reset_index(drop=True); m = rule.mask(g)
        if m.sum() < 60: cells.append(f"{era}: too few"); continue
        e = L.excess_return(g, m, n_boot=800); cells.append(f"{era}: {e['excess_mean']:+.1%} [{e['excess_lo']:+.1%},{e['excess_hi']:+.1%}] n={int(m.sum())}")
    print(f"{rn}\n    " + "   |   ".join(cells))
