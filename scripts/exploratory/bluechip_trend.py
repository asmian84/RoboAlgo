import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from setup_engine import learn as L
import learn_setups as S
RULES = dict(S.TEXTBOOK)
RULES["exploratory_rs_leader"] = L.Rule((L.Condition("rs_W_mom50", ">", 0.80),))
need = {"symbol", "date", "dvol50", "fwd_ret_6m"}
for r in RULES.values(): need |= {c.feature for c in r.conditions}
obs = pd.read_parquet("/Users/asmian/.roboalgo/setups/obs.parquet", columns=sorted(need))
obs = obs[obs.fwd_ret_6m.notna()].copy(); obs["ret"] = obs.fwd_ret_6m; obs["year"] = obs.date.dt.year
sp = pd.read_csv("/Users/asmian/.roboalgo/../RoboAlgo/.claude/worktrees/wonderful-grothendieck-7f313b/../../../../../../Users/asmian/.roboalgo/universe/sp500_2026-10-12.csv") if False else pd.read_csv("/Users/asmian/.roboalgo/universe/sp500_2026-10-12.csv")
sp["sym"] = sp.Symbol.str.replace(".", "-", regex=False); sp["added"] = pd.to_datetime(sp["Date added"], errors="coerce")
added = sp.set_index("sym")["added"]
obs["rank_dv"] = obs.groupby("date").dvol50.rank(ascending=False, method="first")
in_now = obs.symbol.isin(added.index)
obs["added"] = obs.symbol.map(added)
U = {
  "ALL stocks": np.ones(len(obs), bool),
  "TOP-500 by $ volume at the time": (obs.rank_dv <= 500).to_numpy(),
  "today's S&P 500 (hindsight)": in_now.to_numpy(),
  "today's S&P 500, from join date": (in_now & (obs.date >= obs.added.fillna(pd.Timestamp("1900-01-01")))).to_numpy(),
}
eras = {"2001-09": (2001, 2009), "2010-17": (2010, 2017), "2018-26": (2018, 2026)}
print(f"S&P 500 today: {len(sp)} names | in the observation set: {int(sp.sym.isin(obs.symbol.unique()).sum())}\n")
rows = []
for un, um in U.items():
    d_all = obs[um]
    for en, (a, b) in eras.items():
        d = d_all[(d_all.year >= a) & (d_all.year <= b)].reset_index(drop=True)
        if len(d) < 500: continue
        base = float(d.ret.clip(*d.ret.quantile([.01, .99])).mean())
        for rn, rule in RULES.items():
            m = rule.mask(d)
            if m.sum() < 50: rows.append(dict(universe=un, era=en, rule=rn, n=int(m.sum()), note="too few")); continue
            e = L.excess_return(d, m, n_boot=1000)
            rows.append(dict(universe=un, era=en, rule=rn.replace("textbook_", ""), n=int(m.sum()), share=m.mean(), stocks=int(d.symbol[m].nunique()),
                             flagged_ret=float(d.ret[m].clip(*d.ret.quantile([.01, .99])).mean()), universe_ret=base, excess=e["excess_mean"], lo=e["excess_lo"], hi=e["excess_hi"]))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
for rn in R.rule.dropna().unique():
    sub = R[R.rule == rn]
    print(f"=== {rn} : average 6-month return of flagged stocks MINUS the universe's same-date average (95% CI)")
    piv = sub.assign(cell=lambda x: x.apply(lambda r: f"{r.excess:+.1%} [{r.lo:+.1%},{r.hi:+.1%}] n={r.n}" if pd.notna(r.get("excess")) else "too few", axis=1)).pivot(index="universe", columns="era", values="cell")
    print(piv.to_string()); print()
print("universe average 6-month return (winsorised), by era:")
print(R.dropna(subset=["universe_ret"]).drop_duplicates(["universe", "era"]).pivot(index="universe", columns="era", values="universe_ret").map(lambda v: f"{v:+.1%}").to_string())
R.to_csv("/tmp/bluechip_trend_results.csv", index=False)
