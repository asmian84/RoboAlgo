import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from setup_engine import learn as L
import learn_setups as S
RULES = {"stage2_trend": S.TEXTBOOK["textbook_stage2_trend"], "rs_leader (exploratory)": L.Rule((L.Condition("rs_W_mom50", ">", 0.80),))}
need = {"symbol", "date", "dvol50", "fwd_ret_6m"}
for r in RULES.values(): need |= {c.feature for c in r.conditions}
obs = pd.read_parquet("/Users/asmian/.roboalgo/setups/obs.parquet", columns=sorted(need)); obs = obs[obs.fwd_ret_6m.notna()].copy(); obs["ret"] = obs.fwd_ret_6m
obs["rank_dv"] = obs.groupby("date").dvol50.rank(ascending=False, method="first"); bc = obs[obs.rank_dv <= 500].reset_index(drop=True)
spy = pd.read_parquet("/Users/asmian/.roboalgo/benchmarks/SPY.parquet"); spy.index = pd.to_datetime(spy.index)
w = spy.close.resample("W-FRI").last().dropna(); sma = w.rolling(30).mean(); reg = (w > sma).where(sma.notna())
bc["regime_on"] = reg.reindex(bc.date, method="ffill").to_numpy(); bc = bc[bc.regime_on.notna()].copy(); bc["regime_on"] = bc.regime_on.astype(bool); bc = bc.reset_index(drop=True)
bc["era"] = pd.cut(bc.date.dt.year, [2000, 2009, 2017, 2026], labels=["2001-09", "2010-17", "2018-26"])
print("top-500 blue chips at the time; SPY weekly close above its 30-week average = regime ON\n")
wins = lambda s: s.clip(*s.quantile([.01, .99]))
print("share of observation dates with regime ON:", bc.drop_duplicates("date").regime_on.mean().round(2))
t = bc.groupby(["era", "regime_on"], observed=True).apply(lambda g: pd.Series({"avg_6m_return": wins(g.ret).mean(), "share_of_6m_losers(<-10%)": (g.ret < -0.10).mean(), "n": len(g)}))
print(t.round(3).to_string()); print()
for rn, rule in RULES.items():
    print(f"=== {rn}: excess return over the same-date blue-chip average, by regime")
    for on in (True, False):
        for era, g in bc[bc.regime_on == on].groupby("era", observed=True):
            g = g.reset_index(drop=True); m = rule.mask(g)
            if m.sum() < 60: print(f"  regime {'ON ' if on else 'OFF'} {era}: too few ({int(m.sum())})"); continue
            e = L.excess_return(g, m, n_boot=800)
            print(f"  regime {'ON ' if on else 'OFF'} {era}: {e['excess_mean']:+.1%} [{e['excess_lo']:+.1%},{e['excess_hi']:+.1%}]  n={int(m.sum())}")
