"""
Learn chart setups that preceded big runs, and test them on data they were never learned from.

THE PROBLEM THIS IS BUILT AROUND. Pick the stocks that ran, look at their charts beforehand, and
you will always find a "common setup". It proves nothing: the winners are chosen after the fact,
and any pattern, however meaningless, can be found in a few hundred hand-picked winners. A system
that "learns from winners and evolves" without guarding against this is a machine for manufacturing
convincing noise. It is also how the repo's strategy-evolution engine ended up optimising nothing.

So learning here is a procedure with four rules.

  1. PURGED TIME SPLITS. Learn on one era, select on a later one, judge on a later one still. Each
     stage starts only after the previous stage's forward-looking labels have fully resolved, so
     no label the learner trains on describes a period it is later tested on.
  2. LIFT AGAINST THE SAME DATE'S BASE RATE. The share of stocks that ran swings from 0.2% to 7.2%
     depending on the year. A setup is credited only for beating the stocks that were observed on the
     same dates, never for being observed in a good year.
  3. STOCK-LEVEL RESAMPLING. Consecutive observations of one stock are not independent, so confidence
     comes from bootstrapping whole stocks, not observations.
  4. TIMING, NOT TYPE. A rule must beat the SAME STOCK's own base rate, not just the market's. "Volatile,
     beaten-down stocks run more often" is a statement about which stocks, and it is where survivorship
     bias is worst (volatile stocks that crashed were delisted and are missing). A setup is a statement
     about WHEN: this stock, now, is likelier to run than the same stock at other times. The placebo run
     proved the need: on data with the chart-to-outcome link severed but each stock's identity kept, a
     type-only procedure validated 7 of 18 rules.
  5. EVERY ATTEMPT IS LOGGED. Every model and every rule considered goes in an append-only ledger. The
     count of attempts is what the multiple-testing correction needs, and it only ever grows as the
     system evolves.

Known limit, stated here so it travels with the code: the universe is stocks listed TODAY. Companies
that were delisted are absent, so the runners are over-represented and base rates are inflated.
That flatters every number in this module, and a setup that cannot show lift even here is not real.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

@dataclass(frozen=True)
class Objective:
    """What a "runner" is, over what horizon, and how the eras are split. Fixed before any model runs."""
    name: str
    label_col: str          # forward-MAX column defining a runner
    threshold: float        # runner if label_col >= threshold
    ret_col: str            # forward return used for typical-outcome and excess-return checks
    loss_threshold: float   # a "bad outcome" is ret <= this
    horizon_years: float
    splits: dict


# Round one: reaches 5x within three years. Learn -> select -> judge, each stage starting after the
# previous stage's three-year labels have resolved.
SPLITS = {
    "train":    ("2001-01-01", "2011-12-31"),
    "validate": ("2015-01-01", "2017-12-31"),
    "test":     ("2021-01-01", "2023-12-31"),
}
OBJ_3Y = Objective("runner3y", "fwd_max_3y", 4.0, "fwd_ret_3y", -0.5, 3.0, SPLITS)

# Round two: doubles within six months (4.0% of observations; +50% was 13% and too common to mean "runner"). Chosen because survivorship bias shrinks with the horizon: the
# stocks missing from today's listings are those that later died, and a six-month window rarely contains
# much of that slide, so the setup-to-outcome relationship is far less distorted than over three years.
# It is also the horizon the scans are used at. Splits purge six months, not three years.
OBJ_6M = Objective("runner6m", "fwd_max_6m", 1.00, "fwd_ret_6m", -0.30, 0.5, {
    "train":    ("2001-01-01", "2012-12-31"),
    "validate": ("2013-08-01", "2017-12-31"),
    "test":     ("2018-08-01", "2026-03-31"),
})

OBJECTIVES = {o.name: o for o in (OBJ_3Y, OBJ_6M)}
DEFAULT = OBJ_3Y

# kept for callers written against round one
RUNNER_MULTIPLE = 5.0
LABEL_THRESHOLD = RUNNER_MULTIPLE - 1.0
LABEL_HORIZON_YEARS = 3

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"


def check_purged(splits=None, horizon_years=None, obj: Objective = None) -> None:
    """Refuse a split where a stage's labels reach into the next stage's observation window."""
    if obj is not None:
        splits, horizon_years = obj.splits, obj.horizon_years
    splits = SPLITS if splits is None else splits
    horizon_years = LABEL_HORIZON_YEARS if horizon_years is None else horizon_years
    names = list(splits)
    for x, y in zip(names, names[1:]):
        end_x = pd.Timestamp(splits[x][1]) + pd.DateOffset(days=int(round(horizon_years * 365.25)))
        start_y = pd.Timestamp(splits[y][0])
        if end_x > start_y:
            raise ValueError(f"split '{x}' labels run to {end_x.date()}, after '{y}' starts {start_y.date()}: not purged")


for _o in OBJECTIVES.values():
    check_purged(obj=_o)


def feature_columns(obs: pd.DataFrame, timeframes=("D", "W", "M"), with_rs: bool = True) -> list[str]:
    cols = [c for c in obs.columns if c[:2] in tuple(f"{t}_" for t in timeframes)]
    if with_rs:
        cols += [c for c in obs.columns if c.startswith("rs_") and c.split("_", 1)[1][0] in timeframes]
    return cols


def slice_period(obs: pd.DataFrame, name: str, obj: Objective = None) -> pd.DataFrame:
    obj = obj or DEFAULT
    lo, hi = obj.splits[name]
    d = obs[(obs["date"] >= lo) & (obs["date"] <= hi) & obs[obj.label_col].notna() & obs[obj.ret_col].notna()].copy()
    d["runner"] = (d[obj.label_col] >= obj.threshold).astype(int)
    d["ret"] = d[obj.ret_col]                       # the forward return every outcome check reads
    d["bad"] = (d[obj.ret_col] <= obj.loss_threshold).astype(int)
    return d


# ── rules ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Condition:
    feature: str
    op: str            # "<=" or ">"
    value: float

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            return (x <= self.value) if self.op == "<=" else (x > self.value)   # NaN compares False: missing never matches

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.value:.4g}"


@dataclass(frozen=True)
class Rule:
    conditions: tuple

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for c in self.conditions:
            m &= c.mask(df)
        return m

    def __str__(self) -> str:
        return " AND ".join(str(c) for c in self.conditions)

    def to_json(self):
        return [asdict(c) for c in self.conditions]

    @staticmethod
    def from_json(items) -> "Rule":
        return Rule(tuple(Condition(**i) for i in items))


def fit_rules(train: pd.DataFrame, features: list[str], depth: int, min_leaf: int,
              seed: int = 0, target: str = "type_and_timing") -> list[Rule]:
    """Fit a shallow tree on complete cases and return one rule per leaf.

    Complete cases only: a rule must be evaluable on any stock, and "missing" has no honest meaning in
    a printed rule. Models that use monthly bars therefore see only stocks with enough monthly history,
    which is stated wherever such a model's result is reported.
    """
    d = train.dropna(subset=features)
    if d["runner"].sum() < 50:
        return []
    X = d[features].to_numpy(dtype=float)
    if target == "timing":
        # Regress on how much MORE this observation ran than its own stock usually does. Subtracting the
        # stock's mean removes the stock-type effect from the target, so a feature that only identifies
        # a kind of stock has nothing left to explain; what remains is when, within a stock, it ran.
        resid = d["runner"] - d.groupby("symbol")["runner"].transform("mean")
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed).fit(X, resid)
    else:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, class_weight="balanced",
                                      random_state=seed).fit(X, d["runner"])
    t = tree.tree_
    rules: list[Rule] = []

    def walk(node: int, path: tuple):
        if t.children_left[node] == t.children_right[node]:      # a leaf
            if path:
                rules.append(Rule(path))
            return
        f, thr = features[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], path + (Condition(f, "<=", thr),))
        walk(t.children_right[node], path + (Condition(f, ">", thr),))

    walk(0, ())
    return rules


# ── evaluation ─────────────────────────────────────────────────────────────────

def base_rate_by_date(df: pd.DataFrame) -> pd.Series:
    """Share of that date's liquid, labelled stocks that ran. The yardstick for everything."""
    return df.groupby("date")["runner"].mean()


def evaluate(df: pd.DataFrame, mask: np.ndarray, rate_by_date: Optional[pd.Series] = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """How a rule did on a period, against the same dates' base rate, with stock-level uncertainty."""
    if rate_by_date is None:
        rate_by_date = base_rate_by_date(df)
    sub = df.loc[mask, ["symbol", "date", "runner", "ret", "bad"]]
    out = {"n_obs": int(len(sub)), "n_stocks": int(sub["symbol"].nunique()), "n_runners": int(sub["runner"].sum()),
           "base_rate": float(df["runner"].mean())}
    if len(sub) == 0:
        return {**out, "rule_rate": None, "expected": 0.0, "lift": None, "lift_lo": None, "lift_hi": None,
                "p_lift_le_1": None, "timing_lift": None, "timing_lo": None, "timing_hi": None,
                "p_timing_le_1": None, "n_informative_stocks": 0, "median_ret": None, "p_bad": None,
                "excess_mean": None, "excess_lo": None, "excess_hi": None, "p_excess_le_0": None}
    exp = sub["date"].map(rate_by_date).to_numpy()
    # TIMING expectation: what this very stock would be expected to do, given its own run rate over the
    # period and how good that date was for runners generally. A rule is credited only for beating THAT.
    own = df.groupby("symbol")["runner"].mean()
    p_bar = max(float(df["runner"].mean()), 1e-12)
    exp_t = sub["symbol"].map(own).to_numpy() * exp / p_bar
    per = pd.DataFrame({"symbol": sub["symbol"].to_numpy(), "pos": sub["runner"].to_numpy(), "exp": exp, "exp_t": exp_t}).groupby("symbol").sum()
    pos, ex, ext = per["pos"].to_numpy(float), per["exp"].to_numpy(float), per["exp_t"].to_numpy(float)
    lift = pos.sum() / ex.sum() if ex.sum() > 0 else None
    t_lift = pos.sum() / ext.sum() if ext.sum() > 0 else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    boots = pos[idx].sum(axis=1) / np.maximum(ex[idx].sum(axis=1), 1e-12)
    t_boots = pos[idx].sum(axis=1) / np.maximum(ext[idx].sum(axis=1), 1e-12)
    out.update(rule_rate=float(sub["runner"].mean()), expected=float(ex.sum()), lift=float(lift) if lift is not None else None,
               lift_lo=float(np.quantile(boots, 0.025)), lift_hi=float(np.quantile(boots, 0.975)),
               p_lift_le_1=float((boots <= 1.0).mean()),
               timing_lift=float(t_lift) if t_lift is not None else None,
               timing_lo=float(np.quantile(t_boots, 0.025)), timing_hi=float(np.quantile(t_boots, 0.975)),
               p_timing_le_1=float((t_boots <= 1.0).mean()),
               n_informative_stocks=int((own.reindex(per.index) > 0).sum()),
               median_ret=float(sub["ret"].median()), p_bad=float(sub["bad"].mean()))
    out.update(excess_return(df, mask, n_boot=n_boot, seed=seed))
    return out


def excess_return(df: pd.DataFrame, mask: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Average forward return of the flagged observations ABOVE what every stock did on the same date.

    Hitting a big move more often is not the same as making money: round one's rules found 5x runners
    about three times as often and earned no more than the average stock, because the stocks they flagged
    also lost half their value far more often. Returns are winsorised at the 1st/99th percentile so one
    200x name cannot carry the average, and the interval resamples whole stocks.
    """
    lo, hi = df["ret"].quantile([0.01, 0.99])
    w = df["ret"].clip(lo, hi)
    excess = w - w.groupby(df["date"]).transform("mean")
    sub = pd.DataFrame({"symbol": df["symbol"].to_numpy()[mask], "x": excess.to_numpy()[mask]})
    if len(sub) == 0:
        return {"excess_mean": None, "excess_lo": None, "excess_hi": None, "p_excess_le_0": None}
    per = sub.groupby("symbol")["x"].agg(["sum", "count"])
    s_, c_ = per["sum"].to_numpy(float), per["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    boots = s_[idx].sum(axis=1) / np.maximum(c_[idx].sum(axis=1), 1.0)
    return {"excess_mean": float(s_.sum() / c_.sum()), "excess_lo": float(np.quantile(boots, 0.025)),
            "excess_hi": float(np.quantile(boots, 0.975)), "p_excess_le_0": float((boots <= 0.0).mean())}


def lift_within_volatility_terciles(df: pd.DataFrame, mask: np.ndarray, vol_col: str = "W_range10") -> dict:
    """Does the rule beat the base rate among stocks of similar volatility? If its lift vanishes once
    volatility is held fixed, it has found "volatile stocks run more often", which is not a setup."""
    d = df.dropna(subset=[vol_col])
    edges = d[vol_col].quantile([1 / 3, 2 / 3]).to_numpy()
    bucket = np.digitize(df[vol_col].to_numpy(dtype=float), edges)
    res = {}
    for k, name in enumerate(("low_vol", "mid_vol", "high_vol")):
        sel = (bucket == k) & ~np.isnan(df[vol_col].to_numpy(dtype=float))
        sub = df[sel]
        m = mask[sel]
        if m.sum() < 30:
            res[name] = {"n_obs": int(m.sum()), "lift": None}
            continue
        e = evaluate(sub, m, n_boot=300)
        res[name] = {"n_obs": e["n_obs"], "lift": e["timing_lift"]}
    return res


# ── the append-only ledger ─────────────────────────────────────────────────────

def ledger_path() -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    return REGISTRY_DIR / "trials.jsonl"


def log_trial(record: dict) -> None:
    """Append one attempt. Never rewritten, never pruned: its length is the honest trial count."""
    with ledger_path().open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, default=str) + "\n")


def trial_count() -> int:
    p = ledger_path()
    return sum(1 for _ in p.open()) if p.exists() else 0


# ── robustness: would you actually want to be in these? ────────────────────────────
#
# Added AFTER the first learning round, and stated as such. Round one passed its pre-registered
# statistical criteria (22 of 50 rules validated, 0 of 3 placebo runs), and then two diagnostics showed the
# rules were not usable: the flagged stocks mostly collapsed, and the lift vanished in liquid stocks.
# Both are now part of the procedure and apply to every future round as pre-registered criteria. For
# round one they were applied retrospectively, so that verdict is labelled exploratory.

LIQUID_DOLLAR_VOLUME = 10e6          # 50-day average daily traded value
MAX_LOSS_RATIO = 1.5                 # P(bad outcome) may not exceed 1.5x the same era's rate for all stocks
MIN_LIQUID_STOCKS = 40


def robustness(df: pd.DataFrame, mask: np.ndarray, n_boot: int = 1000, primary: str = "timing") -> dict:
    """Three checks a statistical lift alone does not make.

    1. LIQUID STOCKS. Does the timing lift survive among stocks trading >= $10M a day? Illiquid names are
       where survivorship bias is worst (the ones that failed are delisted and gone) and where a rule can
       be profitable on paper but not tradable. A setup that exists only in the illiquid tail is not one.
    2. TYPICAL OUTCOME. A rule can double the odds of a big run and still be a bad idea if what it mostly
       finds is stocks that lose a lot. Lottery tickets are not setups.
    3. MONEY. The flagged stocks must out-earn what every stock earned on the same dates, on winsorised
       returns, with the interval above zero. More hits that earn no more is not an edge.

    `df` must come from `slice_period` so that `ret` and `bad` exist for the objective in use.

    primary="timing" (default) judges a setup that claims to time a run: check 1 asks for timing lift
    among liquid stocks. primary="excess" judges a setup that claims to earn more than the average stock
    (trend and momentum setups, which are LESS likely to double and still out-earn): check 1 asks for a
    positive excess return among liquid stocks instead. Checks 2 and 3 apply to both.
    """
    liquid = (df["dvol50"] >= LIQUID_DOLLAR_VOLUME).to_numpy()
    sub, m = df[liquid], mask[liquid]
    e = evaluate(sub, m, n_boot=n_boot) if m.sum() >= 30 else None
    sel = df[mask]
    bad_rule = float(sel["bad"].mean()) if len(sel) else None
    bad_base = float(df["bad"].mean())
    ex = excess_return(df, mask, n_boot=n_boot)
    reasons = []
    if e is None or e["n_stocks"] < MIN_LIQUID_STOCKS:
        reasons.append("too few liquid stocks to confirm the effect there")
    elif primary == "excess":
        if not (e["excess_lo"] is not None and e["excess_lo"] > 0):
            reasons.append(f"excess return among liquid stocks is {e['excess_mean']:+.1%} [{e['excess_lo']:+.1%}, {e['excess_hi']:+.1%}], interval includes 0")
    elif e["n_informative_stocks"] < MIN_LIQUID_STOCKS:
        reasons.append("too few liquid stocks to confirm the lift there")
    elif not (e["timing_lo"] and e["timing_lo"] > 1.0):
        reasons.append(f"lift among liquid stocks is {e['timing_lift']:.2f} [{e['timing_lo']:.2f}, {e['timing_hi']:.2f}], interval includes 1")
    if bad_rule is not None and bad_rule > MAX_LOSS_RATIO * bad_base:
        reasons.append(f"{bad_rule:.0%} of flagged observations end badly, vs {bad_base:.0%} of all")
    if ex["excess_lo"] is None or ex["excess_lo"] <= 0:
        reasons.append("flagged stocks do not clearly out-earn the same dates' average"
                       + ("" if ex["excess_mean"] is None else f" (excess {ex['excess_mean']:+.1%}, interval [{ex['excess_lo']:+.1%}, {ex['excess_hi']:+.1%}])"))
    return {"passed": not reasons, "reasons": reasons,
            "liquid_timing_lift": e["timing_lift"] if e else None, "liquid_lo": e["timing_lo"] if e else None,
            "liquid_hi": e["timing_hi"] if e else None, "liquid_stocks": e["n_stocks"] if e else 0,
            "liquid_excess_mean": e["excess_mean"] if e else None, "liquid_excess_lo": e["excess_lo"] if e else None,
            "liquid_excess_hi": e["excess_hi"] if e else None,
            "p_bad_rule": bad_rule, "p_bad_all": bad_base,
            "median_ret_rule": float(sel["ret"].median()) if len(sel) else None, "median_ret_all": float(df["ret"].median()),
            "excess_mean": ex["excess_mean"], "excess_lo": ex["excess_lo"], "excess_hi": ex["excess_hi"]}


# ── the registry ───────────────────────────────────────────────────────────────────

def registry_path() -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    return REGISTRY_DIR / "registry.json"


def load_registry() -> dict:
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {"setups": []}


def save_registry(reg: dict) -> None:
    registry_path().write_text(json.dumps(reg, indent=1, default=str))


def scan_current(current: pd.DataFrame, reg: dict, include_unapproved: bool = False) -> pd.DataFrame:
    """Stocks that match registered setups TODAY. Only APPROVED setups are scanned unless asked otherwise:
    a statistical lift that fails the robustness checks must not quietly become a buy list."""
    rows = []
    for s in reg.get("setups", []):
        if s["status"] != "approved" and not include_unapproved:
            continue
        rule = Rule.from_json(s["rule"])
        m = rule.mask(current)
        for _, r in current[m].iterrows():
            rows.append({"symbol": r["symbol"], "setup": s["id"], "status": s["status"], "close": r.get("close"), "dvol50": r.get("dvol50")})
    return pd.DataFrame(rows, columns=["symbol", "setup", "status", "close", "dvol50"])
