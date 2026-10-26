"""
Learn setups that preceded big runs, select them on a later era, judge them once on a later era still.

    python scripts/learn_setups.py --objective runner3y     # round one: 5x within 3 years (reproduces it)
    python scripts/learn_setups.py --objective runner6m     # round two: 2x within 6 months
    python scripts/learn_setups.py --objective runner6m --no-placebo

PRE-REGISTERED. Everything below is fixed before any real result is seen: the objective (label, horizon,
splits), the model grid, every threshold, and the textbook comparators. Changing any of it after looking
at results turns this into a search, and every attempt is appended to the trial ledger
(setup_engine/registry/trials.jsonl) so the count stays honest.

PROCEDURE
  learn     shallow decision trees on the TRAIN era; each leaf is a candidate rule (a setup).
  select    candidates are measured on the VALIDATE era with stocks (not observations) resampled. A rule
            survives only if (a) it beats the SAME STOCK's own base rate, not just the market's ("timing
            lift"), with a confidence interval that clears 1 after correcting for how many candidates were
            tried, (b) it spans enough different stocks, (c) the lift holds inside every volatility
            tercile, and (d) flagged stocks out-earn the same dates' average on winsorised returns.
  judge     survivors are measured ONCE on the TEST era and then put through `learn.robustness`: the lift
            must survive among liquid stocks, flagged stocks must not end badly more than 1.5x as often
            as all stocks, and they must out-earn the same dates' average. Only a setup that passes all
            of it is APPROVED.

PLACEBO. The same procedure is rerun on data where each stock's outcomes have been circularly shifted in
time. That severs any real link between chart and outcome while preserving everything else. A sound
procedure validates roughly nothing there. If it validates several, the procedure is leaking and nothing
it finds on real data can be trusted.

HISTORY OF THE CRITERIA (kept here because it changes what the procedure means):
  * Round one's first placebo run validated 7 of 18 rules, all "which kind of stock" rules, because the
    placebo keeps each stock's identity. The criterion became lift against the same stock's own base rate
    ("timing lift") before any real result was seen.
  * Round one then passed those criteria (22 of 50) and failed two diagnostics added afterwards: flagged
    stocks mostly collapsed, and they earned no more than the average stock. Those became the robustness
    gate and the excess-return criterion, and they are pre-registered for every later round.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import learn as L

OBS = Path.home() / ".roboalgo" / "setups" / "obs.parquet"

# ── pre-registered choices ────────────────────────────────────────────────────
FEATURE_SETS = {"ALL": ("D", "W", "M"), "D": ("D",), "W": ("W",), "M": ("M",)}
DEPTHS = (3, 4)
LEARNER_SETS = {
    "runner3y": {"timing": "timing", "typeTiming": "type_and_timing"},   # round one, kept reproducible
    "runner6m": {"timing": "timing"},                                    # round two: the criterion is timing, so the type learner adds trials and no value
}
MIN_TRAIN_LIFT = 1.3          # TIMING lift: 30% likelier to run than the same stock at other times
MIN_TRAIN_STOCKS = 30         # distinct stocks that have a run at all, so the comparison is informative
MIN_VAL_LIFT = 1.3
MIN_VAL_STOCKS = 25
ALPHA = 0.05                  # family-wise, split across every candidate tried (Bonferroni)
N_BOOT_SELECT = 4000
VOL_COL = "W_range10"

# Textbook setups written by hand BEFORE learning, as comparators: does learning add anything over what
# traders already believe? Measured the same way and counted as trials.
TEXTBOOK = {
    "textbook_stage2_trend": L.Rule((
        L.Condition("W_c_ma30", ">", 0.0), L.Condition("W_ma10_30", ">", 0.0), L.Condition("W_ma30_50", ">", 0.0),
        L.Condition("W_ma50_slope", ">", 0.0), L.Condition("W_dist_hi50", ">", -0.25),
        L.Condition("W_dist_lo50", ">", 0.30), L.Condition("rs_W_mom50", ">", 0.70))),
    "textbook_tight_base_breakout": L.Rule((
        L.Condition("W_ma30_50", ">", 0.0), L.Condition("W_ma50_slope", ">", 0.0),
        L.Condition("W_base_depth30", "<=", 0.35), L.Condition("W_dist_hi50", ">", -0.08),
        L.Condition("W_contraction", "<=", 0.70))),
    "textbook_compression_golden_cross": L.Rule((
        L.Condition("D_ma30_50", ">", 0.0), L.Condition("D_c_ma50", ">", 0.0),
        L.Condition("D_atr_pctl100", "<=", 0.25), L.Condition("D_contraction", "<=", 0.70))),
}


def placebo(obj: L.Objective, obs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Circularly shift each stock's outcomes in time, so features no longer describe what followed.
    Keeps label autocorrelation within a stock and the overall mix of outcomes; drops stocks too short
    to shift beyond the label overlap."""
    rng = np.random.default_rng(seed)
    cols = [c for c in obs.columns if c.startswith("fwd_")]
    parts = []
    for _, g in obs.sort_values(["symbol", "date"]).groupby("symbol"):
        lab = g[[obj.label_col, obj.ret_col]].notna().all(axis=1).to_numpy()
        n = int(lab.sum())
        if n < 100:
            continue
        k = int(rng.integers(45, n - 45))
        g = g.copy()
        g.loc[lab, cols] = np.roll(g.loc[lab, cols].to_numpy(), k, axis=0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def run(obs: pd.DataFrame, obj: L.Objective, tag: str, log: bool) -> dict:
    tr, va, te = (L.slice_period(obs, s, obj) for s in ("train", "validate", "test"))
    base = {"train": tr["runner"].mean(), "validate": va["runner"].mean(), "test": te["runner"].mean()}
    print(f"[{tag}] obs  train {len(tr):,}  validate {len(va):,}  test {len(te):,} | base rate "
          f"{base['train']:.2%} / {base['validate']:.2%} / {base['test']:.2%}", flush=True)
    rate = {"train": L.base_rate_by_date(tr), "validate": L.base_rate_by_date(va), "test": L.base_rate_by_date(te)}

    # ---- learn: every leaf of every model is a candidate
    cands = []
    for fs_name, tfs in FEATURE_SETS.items():
        feats = L.feature_columns(obs, tfs, with_rs=True)
        d = tr.dropna(subset=feats)
        for lname, target in LEARNER_SETS[obj.name].items():
            for depth in DEPTHS:
                min_leaf = max(400, int(0.003 * len(d)))
                rules = L.fit_rules(tr, feats, depth, min_leaf, target=target)
                kept = 0
                for r in rules:
                    e = L.evaluate(tr, r.mask(tr), rate["train"], n_boot=200)
                    if log:
                        L.log_trial({"stage": "train", "run": tag, "objective": obj.name, "model": f"{lname}/{fs_name}/d{depth}",
                                     "rule": str(r), "n_obs": e["n_obs"], "n_stocks": e["n_stocks"], "lift": e["lift"],
                                     "timing_lift": e["timing_lift"]})
                    if e["timing_lift"] and e["timing_lift"] >= MIN_TRAIN_LIFT and e["n_informative_stocks"] >= MIN_TRAIN_STOCKS:
                        cands.append({"model": f"{lname}/{fs_name}/d{depth}", "rule": r, "train": e})
                        kept += 1
                print(f"[{tag}]   {lname:>10} {fs_name:>3}/d{depth}: {len(rules):>2} leaves, {kept} candidate(s) "
                      f"(complete-case train rows {len(d):,})", flush=True)

    comparators = [{"model": "textbook", "name": n, "rule": r} for n, r in TEXTBOOK.items()]
    n_trials = len(cands) + len(comparators)
    print(f"[{tag}] candidates from learning: {len(cands)} | textbook comparators: {len(comparators)} "
          f"| multiple-testing divisor: {n_trials}", flush=True)

    # ---- select on the validation era
    validated, val_rows = [], []
    for c in cands + comparators:
        m = c["rule"].mask(va)
        e = L.evaluate(va, m, rate["validate"], n_boot=N_BOOT_SELECT)
        vol = L.lift_within_volatility_terciles(va, m, VOL_COL)
        p_ok = e["p_timing_le_1"] is not None and e["p_timing_le_1"] <= ALPHA / n_trials
        vol_ok = all(v["lift"] is None or v["lift"] >= 1.0 for v in vol.values()) and any(v["lift"] is not None for v in vol.values())
        money_ok = e["p_excess_le_0"] is not None and e["p_excess_le_0"] <= ALPHA / n_trials
        passed = bool(e["timing_lift"] and e["timing_lift"] >= MIN_VAL_LIFT and e["n_informative_stocks"] >= MIN_VAL_STOCKS
                      and p_ok and vol_ok and money_ok)
        row = {"model": c["model"], "name": c.get("name"), "rule": str(c["rule"]), "validate": e, "vol_terciles": vol,
               "p_ok": p_ok, "vol_ok": vol_ok, "money_ok": money_ok, "passed": passed, "train": c.get("train")}
        val_rows.append(row)
        if log:
            L.log_trial({"stage": "validate", "run": tag, "objective": obj.name, "model": c["model"], "name": c.get("name"),
                         "rule": str(c["rule"]), "lift": e["lift"], "timing_lift": e["timing_lift"], "timing_lo": e["timing_lo"],
                         "excess_mean": e["excess_mean"], "n_stocks": e["n_stocks"], "passed": passed})
        if passed:
            validated.append({**c, "validate": e, "vol_terciles": vol})
    print(f"[{tag}] validated: {len(validated)} of {n_trials}", flush=True)

    # ---- judge once on the test era, then the robustness gate
    test_rows = []
    for c in validated:
        m = c["rule"].mask(te)
        e = L.evaluate(te, m, rate["test"], n_boot=N_BOOT_SELECT)
        vol = L.lift_within_volatility_terciles(te, m, VOL_COL)
        ran = set(te.loc[te["runner"] == 1, "symbol"])
        flagged = set(te.loc[m, "symbol"])
        e.update(recall_stocks=(len(ran & flagged) / len(ran)) if ran else None, n_ran=len(ran),
                 share_of_stocks_flagged=len(flagged) / te["symbol"].nunique())
        rb = L.robustness(te, m, n_boot=N_BOOT_SELECT)
        test_rows.append({"model": c["model"], "name": c.get("name"), "rule": str(c["rule"]), "test": e,
                          "vol_terciles": vol, "robustness": rb})
        if log:
            L.log_trial({"stage": "test", "run": tag, "objective": obj.name, "model": c["model"], "rule": str(c["rule"]),
                         "timing_lift": e["timing_lift"], "timing_lo": e["timing_lo"], "excess_mean": e["excess_mean"],
                         "n_stocks": e["n_stocks"], "robust": rb["passed"], "reasons": rb["reasons"]})

    textbook_test = {}
    for c in comparators:
        textbook_test[c["name"]] = L.evaluate(te, c["rule"].mask(te), rate["test"], n_boot=N_BOOT_SELECT)
    return {"objective": obj.name, "base_rate": base, "n_trials_this_run": n_trials, "candidates": len(cands),
            "validation": val_rows, "validated": [{"model": v["model"], "name": v.get("name"), "rule": str(v["rule"])} for v in validated],
            "test": test_rows, "textbook_on_test": textbook_test}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--objective", choices=sorted(L.OBJECTIVES), default="runner3y")
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--obs", default=str(OBS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    obj = L.OBJECTIVES[args.objective]
    out_path = Path(args.out) if args.out else ROOT / "docs" / (
        "setup_learning_results.json" if obj.name == "runner3y" else f"setup_learning_results_{obj.name}.json")
    obs = pd.read_parquet(args.obs)
    before = L.trial_count()
    real = run(obs, obj, "real", log=True)
    out = {"objective": obj.name, "label": f"{obj.label_col} >= {obj.threshold}", "splits": obj.splits,
           "ledger_trials_before": before, "real": real}
    if not args.no_placebo:
        out["placebo"] = []
        for seed in (1, 2, 3):
            pl = run(placebo(obj, obs, seed), obj, f"placebo{seed}", log=False)
            out["placebo"].append({"seed": seed, **{k: v for k, v in pl.items() if k in ("candidates", "n_trials_this_run", "validated", "base_rate")},
                                   "test": [{"rule": t["rule"], "robust": t["robustness"]["passed"]} for t in pl["test"]]})
    out["ledger_trials_after"] = L.trial_count()
    out_path.write_text(json.dumps(out, indent=1, default=lambda o: None if o is None else str(o)))
    print("wrote", out_path, "| ledger trials:", before, "->", out["ledger_trials_after"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
