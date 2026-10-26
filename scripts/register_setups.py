"""
Turn a learning round into the setup registry, applying the robustness checks.

    python scripts/register_setups.py --objective runner3y     # round one
    python scripts/register_setups.py --objective runner6m     # round two

A setup is APPROVED only if it passes `learn.robustness` on the test era. Everything else is stored with
its status and the reason, so the system remembers WHY a pattern was rejected and does not rediscover it
as new. Re-running for an objective replaces that objective's entries and leaves the others alone.

TWO KINDS OF CLAIM, judged on different evidence:
  * a setup that claims to TIME a run (round one, and the learned rules of round two) needs a timing lift
    that survives among liquid stocks;
  * a TREND setup claims only to earn more than the average stock, and is judged on excess return: the
    interval must clear zero in BOTH the validation and test eras and among liquid stocks. These are the
    hand-written textbook setups, fixed before any learning. Their excess return was first looked at after
    round two's learned rules had failed, so their approval is marked PROVISIONAL: two eras of evidence,
    one of them (2018-2026) much stronger than the other.

Round one's verdicts are exploratory: the robustness checks were added after its test era had been seen.
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
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from setup_engine import learn as L

OBS = Path.home() / ".roboalgo" / "setups" / "obs.parquet"
ROUND = {"runner3y": 1, "runner6m": 2}


def parse(text: str) -> L.Rule:
    return L.Rule(tuple(L.Condition(c.split(" ")[0], c.split(" ")[1], float(c.split(" ")[2])) for c in text.split(" AND ")))


def results_path(obj: L.Objective) -> Path:
    return ROOT / "docs" / ("setup_learning_results.json" if obj.name == "runner3y" else f"setup_learning_results_{obj.name}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--objective", choices=sorted(L.OBJECTIVES), default="runner3y")
    args = ap.parse_args()
    obj = L.OBJECTIVES[args.objective]
    round_no = ROUND[obj.name]
    res = json.loads(results_path(obj).read_text())
    real = res["real"]
    obs = pd.read_parquet(OBS)
    val_df, test_df = L.slice_period(obs, "validate", obj), L.slice_period(obs, "test", obj)

    reg = L.load_registry()
    for s_ in reg["setups"]:
        s_.setdefault("objective", "runner3y")           # round one predates the field
        s_.setdefault("kind", "textbook" if str(s_["id"]).split("-", 1)[-1].startswith("textbook") else "learned")
    setups = [s for s in reg["setups"] if s["objective"] != obj.name]
    rounds = [r for r in reg.get("rounds", []) if r["objective"] != obj.name]

    # ---- learned rules that validated
    by_rule = {}
    for t in real["test"]:
        by_rule.setdefault(t["rule"], {"models": [], "row": t})["models"].append(t["model"])
    valrows = {v["rule"]: v for v in real["validation"]}
    for i, (rule_text, info) in enumerate(sorted(by_rule.items(), key=lambda kv: -(kv[1]["row"]["test"]["timing_lift"] or 0)), 1):
        rule = parse(rule_text)
        rb = info["row"].get("robustness") or L.robustness(test_df, rule.mask(test_df))
        L.log_trial({"stage": "robustness", "objective": obj.name, "rule": rule_text, "passed": rb["passed"], "reasons": rb["reasons"]})
        t = info["row"]["test"]
        setups.append({
            "id": f"r{round_no}-{i:02d}", "round": round_no, "objective": obj.name, "kind": "learned", "rule": rule.to_json(),
            "rule_text": rule_text, "found_by": sorted(set(info["models"])), "label": res["label"],
            "learned_on": obj.splits["train"], "selected_on": obj.splits["validate"], "judged_on": obj.splits["test"],
            "validate": {k: valrows[rule_text]["validate"].get(k) for k in ("timing_lift", "timing_lo", "n_stocks", "excess_mean")},
            "test": {k: t.get(k) for k in ("timing_lift", "timing_lo", "timing_hi", "n_stocks", "excess_mean", "excess_lo", "excess_hi", "rule_rate", "base_rate")},
            "robustness": rb, "status": "approved" if rb["passed"] else "rejected_robustness",
            "rejected_because": rb["reasons"], "exploratory_verdict": round_no == 1})

    # ---- the hand-written textbook setups
    import learn_setups as S
    for name, rule in S.TEXTBOOK.items():
        ev = {era: L.evaluate(df, rule.mask(df), n_boot=2000) for era, df in (("train", L.slice_period(obs, "train", obj)), ("validate", val_df), ("test", test_df))}
        if obj.name == "runner3y":
            e = real["textbook_on_test"][name]
            setups.append({"id": f"r{round_no}-{name}", "round": round_no, "objective": obj.name, "kind": "textbook", "rule_text": name,
                           "rule": rule.to_json(), "status": "rejected_no_edge",
                           "rejected_because": [f"timing lift {e['timing_lift']:.2f} [{e['timing_lo']:.2f}, {e['timing_hi']:.2f}] on the test era: no better than chance, or worse"],
                           "test": {k: e[k] for k in ("timing_lift", "timing_lo", "timing_hi", "n_stocks")}, "exploratory_verdict": False})
            continue
        rb = L.robustness(test_df, rule.mask(test_df), n_boot=2000, primary="excess")
        both = all((ev[era]["excess_lo"] or -1) > 0 for era in ("validate", "test"))
        reasons = list(rb["reasons"]) + ([] if both else ["excess return interval does not clear zero in both the validation and test eras"])
        L.log_trial({"stage": "robustness", "objective": obj.name, "rule": name, "passed": not reasons, "reasons": reasons})
        setups.append({
            "id": f"r{round_no}-{name}", "round": round_no, "objective": obj.name, "kind": "textbook", "rule_text": name,
            "rule": rule.to_json(), "label": f"mean {obj.ret_col} above the same date's average (winsorised 1/99)",
            "excess_by_era": {era: {k: ev[era][k] for k in ("excess_mean", "excess_lo", "excess_hi", "n_obs", "n_stocks", "timing_lift", "timing_lo", "timing_hi")} for era in ev},
            "robustness": rb, "status": "approved" if not reasons else "rejected_robustness", "rejected_because": reasons,
            "provisional": True,
            "caveat": ("Earns more than the average stock but is LESS likely to double: its timing lift is below 1. Excess return is "
                       "measured on 6-month winsorised returns before costs, against the equal-weighted average of liquid listed stocks. "
                       "Two eras of evidence; 2001-2012 shows about +1% per half-year, 2018-2026 about +2% to +5%."),
            "exploratory_verdict": True})

    n_cand = real["candidates"]
    rounds.append({"round": round_no, "objective": obj.name, "label": res["label"], "learned_candidates": n_cand,
                   "learned_validated": len(real["validated"]), "placebo_validated": [len(p["validated"]) for p in res.get("placebo", [])]})
    reg = {"label": "see rounds", "trial_count": L.trial_count(), "rounds": sorted(rounds, key=lambda r: r["round"]),
           "setups": sorted(setups, key=lambda s: (s["round"], s["id"]))}
    L.save_registry(reg)
    mine = [s for s in setups if s["objective"] == obj.name]
    print(f"round {round_no} ({obj.name}): registered {len(mine)} | approved {sum(s['status'] == 'approved' for s in mine)} "
          f"| learned validated {len(real['validated'])} of {n_cand} candidates | ledger trials {reg['trial_count']}")
    for s in mine:
        if s["kind"] == "textbook" and s["round"] == round_no and obj.name == "runner6m":
            e = s["excess_by_era"]
            print("   %-38s %-19s excess train %+.1f%% | validate %+.1f%% [%+.1f%%,%+.1f%%] | test %+.1f%% [%+.1f%%,%+.1f%%] | liquid excess %s" % (
                s["rule_text"], s["status"], e["train"]["excess_mean"] * 100, e["validate"]["excess_mean"] * 100, e["validate"]["excess_lo"] * 100,
                e["validate"]["excess_hi"] * 100, e["test"]["excess_mean"] * 100, e["test"]["excess_lo"] * 100, e["test"]["excess_hi"] * 100,
                "n/a" if s["robustness"]["liquid_excess_mean"] is None else f"{s['robustness']['liquid_excess_mean']:+.1%} [{s['robustness']['liquid_excess_lo']:+.1%},{s['robustness']['liquid_excess_hi']:+.1%}]"))
            if s["rejected_because"]:
                print("        rejected because:", "; ".join(s["rejected_because"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
