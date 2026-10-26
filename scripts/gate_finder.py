"""
Run the Finder's replay results through the validation gate.

    python scripts/gate_finder.py                       # reads docs/finder_validation_results.json

Each preset is a CANDIDATE, judged against the gate's default limits (OOS Sharpe >= 1.0,
drawdown <= 15%, hit rate >= 55%, >= 30 trades, deflated Sharpe >= 0.95).

The trial ledger is every configuration that scripts/validate_finder.py evaluated, across
all groups. Each candidate is deflated against ALL of them, because choosing the best
preset or hold period out of that set is the selection the correction exists for. The
ledger holds the OOS annualized Sharpe of each configuration, measured the same way as
the candidate (uncapped with uncapped, top-10 with top-10).

Two portfolio types are judged separately, and BOTH are shown:
  uncapped: every signal taken. The whole-universe edge test.
  top10:    10 positions in the Finder's own ranking. What can actually be traded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_validation.gates import PerformanceStats, StrategyCandidate, ValidationLimits, evaluate

CANDIDATES = ["signal.base", "pennyking.base", "maker.base", "penny_floor1.base"]


def stats(d: dict) -> PerformanceStats:
    return PerformanceStats(
        sharpe=d["sharpe"] if d["sharpe"] is not None else 0.0,
        max_drawdown_pct=d["max_drawdown_pct"] if d["max_drawdown_pct"] is not None else 0.0,
        hit_rate=d["hit_rate"] if d["hit_rate"] is not None else 0.0,
        n_trades=d["n_trades"], n_obs=d["n_obs"], skew=d["skew"], kurtosis=d["kurtosis"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default=str(ROOT / "docs" / "finder_validation_results.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "finder_gate_verdicts.json"))
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    configs = {}
    for g in res["groups"].values():
        configs.update(g["configs"])
    n_trials = len(configs)
    print(f"trial ledger: {n_trials} configurations -> {sorted(configs)}\n")

    verdicts = {}
    for kind in ("uncapped", "top10"):
        ledger = [(c[kind]["out_of_sample"]["sharpe"] or 0.0) for c in configs.values()]
        print(f"=== {kind.upper()} portfolio | ledger OOS Sharpes: "
              f"min {min(ledger):.2f}  median {sorted(ledger)[len(ledger)//2]:.2f}  max {max(ledger):.2f}")
        for name in CANDIDATES:
            if name not in configs:
                continue
            c = configs[name][kind]
            cand = StrategyCandidate(in_sample=stats(c["in_sample"]), out_of_sample=stats(c["out_of_sample"]),
                                     n_trials=n_trials, name=f"{name} [{kind}]", trial_sharpes=ledger)
            v = evaluate(cand, ValidationLimits())
            o, i = c["out_of_sample"], c["in_sample"]
            print(f"  {name:<18} {'APPROVED' if v.approved else 'REJECTED':<9} reason={v.reason:<22}"
                  f" | OOS Sharpe {o['sharpe'] or 0:5.2f}  DD {o['max_drawdown_pct'] or 0:5.1f}%  hit {o['hit_rate'] or 0:.2f}"
                  f"  trades {o['n_trades']:>6} | IS Sharpe {i['sharpe'] or 0:5.2f}")
            for k in ("dsr", "benchmark_sharpe", "deflated_sharpe", "sr0", "oos_is_ratio"):
                if k in v.metrics:
                    print(f"      {k} = {v.metrics[k]}")
            verdicts[f"{name}|{kind}"] = {"approved": v.approved, "reason": v.reason,
                                          "metrics": {k: (x if isinstance(x, (int, float, str, bool, type(None))) else str(x))
                                                      for k, x in v.metrics.items()}}
        print()
    Path(args.out).write_text(json.dumps({"n_trials": n_trials, "limits": ValidationLimits().__dict__,
                                          "verdicts": verdicts}, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
