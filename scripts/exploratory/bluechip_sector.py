"""EXPLORATORY, UNTESTED: sector-momentum rotation inside today's S&P 500 members, counted only from their join dates.
Sectors are the CURRENT GICS labels. Reads ~/.roboalgo/setups/obs.parquet and ~/.roboalgo/universe/sp500_*.csv. Result reported in docs/BLUECHIP_STUDY_2026-10-12.md."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, pandas as pd
obs = pd.read_parquet("/Users/asmian/.roboalgo/setups/obs.parquet", columns=["symbol", "date", "dvol50", "W_mom30", "fwd_ret_6m"])
obs = obs[obs.fwd_ret_6m.notna() & obs.W_mom30.notna()].copy()
sp = pd.read_csv(sorted(glob.glob("/Users/asmian/.roboalgo/universe/sp500_*.csv"))[-1])
sp["sym"] = sp.Symbol.str.replace(".", "-", regex=False); sp["added"] = pd.to_datetime(sp["Date added"], errors="coerce")
d = obs.join(sp.set_index("sym")[["GICS Sector", "added"]], on="symbol", how="inner")
d = d[d.date >= d.added.fillna(pd.Timestamp("1900-01-01"))].rename(columns={"GICS Sector": "sector"})
d["ret"] = d.fwd_ret_6m.clip(*d.fwd_ret_6m.quantile([.01, .99]))
g = d.groupby(["date", "sector"]).agg(mom=("W_mom30", "median"), n=("symbol", "size"), fwd=("ret", "mean")).reset_index()
g = g[g.n >= 8]; g["rank"] = g.groupby("date").mom.rank(ascending=False, method="first"); g["k"] = g.groupby("date").mom.transform("size"); g = g[g.k >= 8]
g["grp"] = np.where(g["rank"] <= 3, "top-3 sectors", np.where(g["rank"] > g.k - 3, "bottom-3 sectors", "middle"))
g["ex"] = g.fwd - g.groupby("date").fwd.transform("mean"); g["era"] = pd.cut(g.date.dt.year, [2000, 2009, 2017, 2026], labels=["2001-09", "2010-17", "2018-26"])
def spread(sub):
    p = sub.pivot_table(index="date", columns="grp", values="ex", aggfunc="mean").dropna(subset=["top-3 sectors", "bottom-3 sectors"])
    s = (p["top-3 sectors"] - p["bottom-3 sectors"]).to_numpy(); rng = np.random.default_rng(0); b = [rng.choice(s, len(s)).mean() for _ in range(2000)]
    return s.mean(), np.quantile(b, .025), np.quantile(b, .975), len(s)
for era, sub in g.groupby("era", observed=True):
    mn, lo, hi, n = spread(sub); print(f"{era}: top minus bottom sectors {mn:+.1%} [{lo:+.1%},{hi:+.1%}] ({n} dates)")
