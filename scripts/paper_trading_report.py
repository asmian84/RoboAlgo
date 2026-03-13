#!/usr/bin/env python3
"""
RoboAlgo — Paper Trading Report
CLI script to run and display paper trading simulation results.

Usage:
  # 1-week simulation
  python scripts/paper_trading_report.py --period 1w

  # 1-month simulation
  python scripts/paper_trading_report.py --period 1m

  # Custom date range
  python scripts/paper_trading_report.py --start 2025-01-01 --end 2025-03-01

  # Just show existing results (no new simulation)
  python scripts/paper_trading_report.py --report-only

  # Reset and re-run
  python scripts/paper_trading_report.py --period 1m --reset
"""

import argparse
import sys
import os
from datetime import date, timedelta

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_engine.trader import PaperTrader


# ── ANSI colour helpers ─────────────────────────────────────────────────────

def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def yellow(s):return f"\033[93m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"
def dim(s):   return f"\033[2m{s}\033[0m"
def cyan(s):  return f"\033[96m{s}\033[0m"


def color_pnl(val: float) -> str:
    s = f"${val:+,.2f}" if val != 0 else "$0.00"
    return green(s) if val >= 0 else red(s)

def color_pct(val: float) -> str:
    s = f"{val:+.2f}%"
    return green(s) if val >= 0 else red(s)


# ── Report renderer ─────────────────────────────────────────────────────────

def print_report(summary: dict, start_date: date, end_date: date):
    trades     = summary.get("trades", [])
    snapshots  = summary.get("daily_equity", [])

    sep = "─" * 52

    print()
    print(bold(cyan("  ╔══════════════════════════════════════════════════╗")))
    print(bold(cyan("  ║         ROBOALGO  PAPER TRADING REPORT          ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════════════╝")))
    print()
    print(f"  Period : {start_date}  →  {end_date}")
    print(f"  Days   : {(end_date - start_date).days + 1}")
    print()
    print(f"  {sep}")
    print(bold("  ACCOUNT SUMMARY"))
    print(f"  {sep}")
    print(f"  Starting Balance  : ${summary['starting_balance']:>12,.2f}")
    print(f"  Ending Balance    : ${summary['ending_balance']:>12,.2f}   {color_pct(summary['total_return_pct'])}")
    print(f"  Total P&L         : {color_pnl(summary['total_pnl'])}")
    print(f"  Max Drawdown      : {red(f\"{summary['max_drawdown_pct']:.2f}%\")}")
    print()
    print(f"  {sep}")
    print(bold("  TRADE STATISTICS"))
    print(f"  {sep}")
    print(f"  Trades Taken      : {summary['number_of_trades']}")

    if summary['number_of_trades'] > 0:
        wr_str  = f"{summary['win_rate']:.1f}%"
        wr_col  = green(wr_str) if summary['win_rate'] >= 50 else red(wr_str)
        avg_str = color_pct(summary['average_return_pct'])
        print(f"  Win Rate          : {wr_col}")
        print(f"  Average Return    : {avg_str}")
        print(f"  Largest Win       : {color_pnl(summary['largest_win'])}")
        print(f"  Largest Loss      : {color_pnl(summary['largest_loss'])}")

    print()
    print(f"  Open Positions    : {summary['open_positions']}")

    # ── Equity curve (mini sparkline) ──────────────────────────────────────
    if snapshots and len(snapshots) >= 2:
        equity = [s["ending_balance"] for s in snapshots]
        lo, hi = min(equity), max(equity)
        bars  = "▁▂▃▄▅▆▇█"
        if hi > lo:
            spark = "".join(bars[int((v - lo) / (hi - lo) * 7)] for v in equity)
        else:
            spark = "─" * len(equity)
        print()
        print(f"  {sep}")
        print(bold("  EQUITY CURVE"))
        print(f"  {sep}")
        col = green if equity[-1] >= equity[0] else red
        print(f"  {col(spark[:48])}")
        print(f"  ${lo:,.0f}  →  ${hi:,.0f}")

    # ── Recent trades ──────────────────────────────────────────────────────
    if trades:
        print()
        print(f"  {sep}")
        print(bold("  TRADE LOG  (most recent first)"))
        print(f"  {sep}")
        header = f"  {'Symbol':<8} {'Entry':>10} {'Exit':>10} {'Shares':>6} {'PnL':>10} {'Return':>8} {'Reason':<16}"
        print(dim(header))

        for t in trades[:20]:
            pnl_str = color_pnl(t['pnl'])
            ret_str = color_pct(t['return_percent'])
            reason  = t.get('exit_reason', '—')
            reason_col = (green if reason == 'target' else red if reason == 'stop' else yellow)(reason)
            line = (
                f"  {t['symbol']:<8} {t['entry_date']:>10} {t['exit_date']:>10} "
                f"{t['position_size']:>6}  {pnl_str:>10}  {ret_str:>8}  {reason_col}"
            )
            print(line)

        if len(trades) > 20:
            print(f"  {dim(f'... and {len(trades)-20} more trades')}")

    print()
    print(f"  {sep}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RoboAlgo Paper Trading Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--period", choices=["1w", "1m", "3m", "6m", "1y"],
        help="Simulation period shortcut: 1w=1 week, 1m=1 month, 3m=3 months, etc.",
    )
    parser.add_argument("--start", type=str, help="Custom start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, help="Custom end date YYYY-MM-DD")
    parser.add_argument("--reset", action="store_true", help="Clear existing data before running")
    parser.add_argument("--report-only", action="store_true", help="Show existing results without running simulation")
    return parser.parse_args()


def resolve_dates(args) -> tuple[date, date]:
    today = date.today()

    if args.period:
        deltas = {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days   = deltas[args.period]
        return today - timedelta(days=days), today

    if args.start and args.end:
        return (
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )

    # Default: last 30 days
    return today - timedelta(days=30), today


def main():
    args = parse_args()
    start_date, end_date = resolve_dates(args)

    trader = PaperTrader()
    try:
        if args.report_only:
            print(f"\n  Loading existing results...")
            summary = trader.get_summary(start_date=start_date, end_date=end_date)
        else:
            period_label = args.period or f"{start_date} to {end_date}"
            print(f"\n  Running paper simulation: {period_label}  (reset={args.reset})")
            summary = trader.run_simulation(
                start_date=start_date,
                end_date=end_date,
                reset=args.reset,
            )

        print_report(summary, start_date, end_date)

    finally:
        trader.close()


if __name__ == "__main__":
    main()
