#!/usr/bin/env python3
"""
RoboAlgo — Instrument Seeder
Populates the instruments table with a comprehensive set of ETFs, sector ETFs,
leveraged ETFs, commodity ETFs, and volatility instruments.

Usage:
    python scripts/seed_instruments.py [--dry-run] [--clear]

Options:
    --dry-run   Print what would be inserted without writing to DB.
    --clear     Remove all existing instruments before seeding (CAREFUL).
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_session
from database.models import Instrument

# ── Instrument definitions ───────────────────────────────────────────────────
# Each entry: (symbol, name, instrument_type, leverage_factor, underlying, pair_symbol)

INSTRUMENTS: list[tuple] = [
    # ── Index ETFs (unleveraged) ─────────────────────────────────────────────
    ("SPY",  "SPDR S&P 500 ETF Trust",                  "index",       1.0,  "S&P 500",          None),
    ("QQQ",  "Invesco QQQ Trust (Nasdaq-100)",           "index",       1.0,  "Nasdaq-100",       None),
    ("IWM",  "iShares Russell 2000 ETF",                 "index",       1.0,  "Russell 2000",     None),
    ("DIA",  "SPDR Dow Jones Industrial Average ETF",    "index",       1.0,  "DJIA",             None),
    ("VTI",  "Vanguard Total Stock Market ETF",          "index",       1.0,  "Total Market",     None),
    ("MDY",  "SPDR S&P MidCap 400 ETF",                  "index",       1.0,  "S&P MidCap 400",  None),
    ("IVV",  "iShares Core S&P 500 ETF",                 "index",       1.0,  "S&P 500",          None),
    ("VOO",  "Vanguard S&P 500 ETF",                     "index",       1.0,  "S&P 500",          None),

    # ── 3× Bull Leveraged Index ETFs ─────────────────────────────────────────
    ("TQQQ", "ProShares UltraPro QQQ",                   "leveraged_etf_bull", 3.0, "Nasdaq-100",  "SQQQ"),
    ("UPRO", "ProShares UltraPro S&P 500",               "leveraged_etf_bull", 3.0, "S&P 500",     "SPXS"),
    ("TNA",  "Direxion Daily Small Cap Bull 3X Shares",  "leveraged_etf_bull", 3.0, "Russell 2000","TZA"),
    ("UDOW", "ProShares UltraPro Dow 30",                 "leveraged_etf_bull", 3.0, "DJIA",        "SDOW"),
    ("LABU", "Direxion Daily S&P Biotech Bull 3X",        "leveraged_etf_bull", 3.0, "S&P Biotech", "LABD"),
    ("SOXL", "Direxion Daily Semiconductors Bull 3X",     "leveraged_etf_bull", 3.0, "PHLX SOX",   "SOXS"),
    ("TECL", "Direxion Daily Technology Bull 3X",         "leveraged_etf_bull", 3.0, "Technology",  "TECS"),
    ("FNGU", "MicroSectors FANG+ Index 3X Leveraged",    "leveraged_etf_bull", 3.0, "FANG+",       "FNGD"),
    ("TSLL", "Direxion Daily TSLA Bull 1.5X",             "leveraged_etf_bull", 1.5,"Tesla",        "TSLS"),
    ("NVDL", "GraniteShares 2x Long NVDA Daily ETF",      "leveraged_etf_bull", 2.0,"NVIDIA",       "NVDS"),
    ("MSFO", "T-Rex 2X Long MSFT Daily ETF",              "leveraged_etf_bull", 2.0,"Microsoft",    None),
    ("AMZU", "Direxion Daily AMZN Bull 1.5X",             "leveraged_etf_bull", 1.5,"Amazon",       None),
    ("FAS",  "Direxion Daily Financial Bull 3X Shares",   "leveraged_etf_bull", 3.0, "Financial",   "FAZ"),
    ("ERX",  "Direxion Daily Energy Bull 2X Shares",      "leveraged_etf_bull", 2.0, "Energy",      "ERY"),

    # ── 3× Bear Leveraged Index ETFs ─────────────────────────────────────────
    ("SQQQ", "ProShares UltraPro Short QQQ",             "leveraged_etf_bear", -3.0, "Nasdaq-100",  "TQQQ"),
    ("SPXS", "Direxion Daily S&P 500 Bear 3X Shares",    "leveraged_etf_bear", -3.0, "S&P 500",     "UPRO"),
    ("TZA",  "Direxion Daily Small Cap Bear 3X Shares",  "leveraged_etf_bear", -3.0, "Russell 2000","TNA"),
    ("SDOW", "ProShares UltraPro Short Dow 30",          "leveraged_etf_bear", -3.0, "DJIA",        "UDOW"),
    ("SOXS", "Direxion Daily Semiconductors Bear 3X",    "leveraged_etf_bear", -3.0, "PHLX SOX",   "SOXL"),
    ("TECS", "Direxion Daily Technology Bear 3X",         "leveraged_etf_bear", -3.0, "Technology",  "TECL"),
    ("LABD", "Direxion Daily S&P Biotech Bear 3X",        "leveraged_etf_bear", -3.0, "S&P Biotech","LABU"),
    ("FAZ",  "Direxion Daily Financial Bear 3X Shares",  "leveraged_etf_bear", -3.0, "Financial",   "FAS"),
    ("FNGD", "MicroSectors FANG+ Index -3X Inverse",     "leveraged_etf_bear", -3.0, "FANG+",       "FNGU"),

    # ── Sector ETFs (unleveraged) ────────────────────────────────────────────
    ("XLK",  "Technology Select Sector SPDR",             "index",       1.0, "Technology",       None),
    ("XLF",  "Financial Select Sector SPDR",              "index",       1.0, "Financials",       None),
    ("XLE",  "Energy Select Sector SPDR",                 "index",       1.0, "Energy",           None),
    ("XLV",  "Health Care Select Sector SPDR",            "index",       1.0, "Health Care",      None),
    ("XLI",  "Industrial Select Sector SPDR",             "index",       1.0, "Industrials",      None),
    ("XLC",  "Communication Services Select Sector SPDR","index",       1.0, "Comm Services",    None),
    ("XLY",  "Consumer Discretionary Select Sector SPDR","index",       1.0, "Cons Discr",       None),
    ("XLP",  "Consumer Staples Select Sector SPDR",       "index",       1.0, "Cons Staples",     None),
    ("XLB",  "Materials Select Sector SPDR",              "index",       1.0, "Materials",        None),
    ("XLRE", "Real Estate Select Sector SPDR",            "index",       1.0, "Real Estate",      None),
    ("XLU",  "Utilities Select Sector SPDR",              "index",       1.0, "Utilities",        None),

    # ── Commodity ETFs ────────────────────────────────────────────────────────
    ("GLD",  "SPDR Gold Shares",                          "commodity",   1.0, "Gold",             None),
    ("SLV",  "iShares Silver Trust",                      "commodity",   1.0, "Silver",           None),
    ("USO",  "United States Oil Fund",                    "commodity",   1.0, "WTI Crude Oil",    None),
    ("UNG",  "United States Natural Gas Fund",            "commodity",   1.0, "Natural Gas",      None),
    ("PDBC", "Invesco Optimum Yield Diversified Commodity","commodity",  1.0, "Diversified Comm.",None),
    ("IAU",  "iShares Gold Trust",                        "commodity",   1.0, "Gold",             None),

    # ── Volatility ────────────────────────────────────────────────────────────
    ("VXX",  "iPath Series B S&P 500 VIX Short-Term Futures ETN","index",1.0,"VIX Short-Term",   None),
    ("UVXY", "ProShares Ultra VIX Short-Term Futures ETF",       "leveraged_etf_bull",1.5,"VIX",  "SVXY"),
    ("SVXY", "ProShares Short VIX Short-Term Futures ETF",       "leveraged_etf_bear",-0.5,"VIX", "UVXY"),
    ("VIXY", "ProShares VIX Short-Term Futures ETF",             "index",1.0,"VIX Short-Term",   None),

    # ── Popular large-cap stocks ─────────────────────────────────────────────
    ("AAPL", "Apple Inc.",                                "stock",       1.0, None,               None),
    ("MSFT", "Microsoft Corporation",                     "stock",       1.0, None,               None),
    ("NVDA", "NVIDIA Corporation",                        "stock",       1.0, None,               None),
    ("AMZN", "Amazon.com Inc.",                           "stock",       1.0, None,               None),
    ("GOOGL","Alphabet Inc. Class A",                     "stock",       1.0, None,               None),
    ("META", "Meta Platforms Inc.",                       "stock",       1.0, None,               None),
    ("TSLA", "Tesla Inc.",                                "stock",       1.0, None,               None),
    ("AMD",  "Advanced Micro Devices Inc.",               "stock",       1.0, None,               None),
    ("MSTR", "MicroStrategy Incorporated",               "stock",       1.0, None,               None),
    ("COIN", "Coinbase Global Inc.",                      "stock",       1.0, None,               None),
    ("SMCI", "Super Micro Computer Inc.",                 "stock",       1.0, None,               None),
    ("PLTR", "Palantir Technologies Inc.",                "stock",       1.0, None,               None),
    ("ARM",  "Arm Holdings plc",                          "stock",       1.0, None,               None),
    ("AVGO", "Broadcom Inc.",                             "stock",       1.0, None,               None),
    ("CRM",  "Salesforce Inc.",                           "stock",       1.0, None,               None),
    ("TSM",  "Taiwan Semiconductor Mfg Co Ltd (ADR)",     "stock",       1.0, None,               None),
]


def seed(dry_run: bool = False, clear: bool = False) -> None:
    added   = 0
    updated = 0
    skipped = 0

    print(f"\n{'DRY RUN — ' if dry_run else ''}Seeding {len(INSTRUMENTS)} instruments…\n")

    with get_session() as session:
        if clear and not dry_run:
            count = session.query(Instrument).count()
            session.query(Instrument).delete()
            session.commit()
            print(f"  ⚠  Cleared {count} existing instruments.\n")

        for symbol, name, inst_type, lev, underlying, pair in INSTRUMENTS:
            existing = session.execute(
                __import__('sqlalchemy').select(Instrument).where(Instrument.symbol == symbol)
            ).scalar_one_or_none()

            if dry_run:
                status = "UPDATE" if existing else "INSERT"
                print(f"  [{status}] {symbol:6s}  {name[:45]:45s}  {inst_type:20s}  lev={lev}")
                continue

            if existing:
                # Update metadata if it has changed
                changed = False
                if existing.name != name:               existing.name = name;               changed = True
                if existing.instrument_type != inst_type: existing.instrument_type = inst_type; changed = True
                if existing.leverage_factor != lev:     existing.leverage_factor = lev;     changed = True
                if existing.underlying != underlying:   existing.underlying = underlying;   changed = True
                if existing.pair_symbol != pair:        existing.pair_symbol = pair;        changed = True
                if changed:
                    updated += 1
                    print(f"  ✎  Updated:  {symbol}")
                else:
                    skipped += 1
            else:
                inst = Instrument(
                    symbol          = symbol,
                    name            = name,
                    instrument_type = inst_type,
                    leverage_factor = lev,
                    underlying      = underlying,
                    pair_symbol     = pair,
                )
                session.add(inst)
                added += 1
                print(f"  ✚  Inserted: {symbol}  ({name[:40]})")

        if not dry_run:
            session.commit()
            print(f"\n✅ Done — {added} inserted, {updated} updated, {skipped} unchanged\n")
        else:
            print(f"\n[DRY RUN] Would insert/update {len(INSTRUMENTS)} instruments.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed instrument catalogue")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without writing")
    parser.add_argument("--clear",   action="store_true", help="Delete all instruments first")
    args = parser.parse_args()
    seed(dry_run=args.dry_run, clear=args.clear)
