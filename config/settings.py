"""
RoboAlgo - Configuration
Central configuration for database, instruments, and system parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://asmian@localhost:5432/roboalgo")

# ── Instrument Universe ──────────────────────────────────────────────────

# Index drivers (benchmarks)
INDEX_DRIVERS = ["QQQ", "SPY", "SOXX", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV"]

# Index leveraged ETFs: (bull, bear, description, underlying)
INDEX_LEVERAGED = [
    ("TQQQ", "SQQQ", "NASDAQ-100 3x", "QQQ"),
    ("UPRO", "SPXU", "S&P 500 3x", "SPY"),
    ("SSO",  "SDS",  "S&P 500 2x", "SPY"),
    ("UDOW", "SDOW", "Dow Jones 3x", "SPY"),
    ("TNA",  "TZA",  "Russell 2000 3x", "IWM"),
    ("WANT", None,   "Consumer Disc 3x", "XLY"),
]

# Sector leveraged ETFs
SECTOR_LEVERAGED = [
    ("SOXL", "SOXS", "Semiconductor 3x", "SOXX"),
    ("FAS",  "FAZ",  "Financial 3x", "JPM"),
    ("TECL", "TECS", "Technology 3x", "QQQ"),
    ("CURE", None,   "Healthcare 3x", "XLV"),
    ("DFEN", None,   "Aerospace & Defense 3x", "ITA"),
    ("MIDU", None,   "Mid-Cap 3x", "MDY"),
    ("LABU", "LABD", "Biotech 3x", "XBI"),
    ("NAIL", None,   "Homebuilders 3x", "ITB"),
    ("WEBL", "WEBS", "Online Retail 3x", "FDN"),
    ("RETL", None,   "Retail 3x", "XRT"),
]

# Commodity leveraged ETFs
COMMODITY_LEVERAGED = [
    ("GUSH", "DRIP", "Oil & Gas E&P 2x", None),
    ("BOIL", "KOLD", "Natural Gas 2x", None),
    ("UGL",  "GLL",  "Gold 2x", None),
    ("UCO",  "SCO",  "Crude Oil 2x", None),
    ("AGQ",  "ZSL",  "Silver 2x", None),
]

# Single-stock leveraged ETFs: (bull, bear_or_None, description, underlying)
SINGLE_STOCK_LEVERAGED = [
    # ── Mag-7 / mega-cap (original) ─────────────────────────────────
    ("MSTU",  "MSTZ",  "MicroStrategy 2x (T-Rex / T-Rex Inverse)",    "MSTR"),
    ("NVDL",  "NVDS",  "NVIDIA 2x (GraniteShares / Tradr Short)",      "NVDA"),
    ("NVDU",  "NVD",   "NVIDIA 2x (Direxion / GraniteShares Short)",   "NVDA"),
    ("TSLL",  "TSLQ",  "Tesla 2x (Direxion / Tradr Short)",            "TSLA"),
    ("AAPU",  "AAPD",  "Apple 2x (Direxion)",                          "AAPL"),
    ("AMZU",  "AMZD",  "Amazon 2x (Direxion)",                         "AMZN"),
    ("METU",  "METD",  "Meta 2x (Direxion)",                           "META"),
    ("GGLL",  "GGLS",  "Google 2x (Direxion)",                         "GOOGL"),
    ("MSFU",  "MSFD",  "Microsoft 2x (Direxion)",                      "MSFT"),
    ("AMDL",  "AMDD",  "AMD 2x (GraniteShares)",                       "AMD"),
    # ── Semi / tech ─────────────────────────────────────────────────
    ("PLTU",  "PLTZ",  "Palantir 2x (Direxion / Defiance Short)",      "PLTR"),
    ("MUU",   "MULL",  "Micron 2x (Direxion / GraniteShares)",         "MU"),
    ("SMCX",  "SMST",  "Super Micro 2x (Defiance / Defiance Short)",   "SMCI"),
    ("IONX",  "IONL",  "IonQ 2x (Defiance / GraniteShares)",           "IONQ"),
    ("AVGX",  "AVL",   "Broadcom 2x (Defiance / Direxion)",            "AVGO"),
    ("ORCX",  None,    "Oracle 2x (Defiance)",                         "ORCL"),
    ("MSFL",  None,    "Microsoft 2x (GraniteShares)",                  "MSFT"),
    ("AMDG",  None,    "AMD 2x (Leverage Shares)",                     "AMD"),
    ("NFXL",  None,    "Netflix 2x (Direxion)",                        "NFLX"),
    # ── Fin / crypto ─────────────────────────────────────────────────
    ("CONL",  None,    "Coinbase 2x (GraniteShares)",                  "COIN"),
    ("MRAL",  None,    "Marathon Digital 2x (GraniteShares)",          "MARA"),
    ("BRKU",  None,    "Berkshire B 2x (Direxion)",                    "BRK-B"),
    ("BABX",  None,    "Alibaba 2x (GraniteShares)",                   "BABA"),
    ("RDTL",  None,    "Reddit 2x (GraniteShares)",                    "RDDT"),
    # ── High-momentum growth ─────────────────────────────────────────
    ("TSMU",  "TSMG",  "TSMC 2x (GraniteShares / Leverage Shares)",   "TSM"),
    ("CRWL",  None,    "CrowdStrike 2x (GraniteShares)",               "CRWD"),
    ("SOFX",  None,    "SoFi 2x (Defiance)",                           "SOFI"),
    ("RKLX",  None,    "Rocket Lab 2x (Defiance)",                     "RKLB"),
    ("UBRL",  None,    "Uber 2x (GraniteShares)",                      "UBER"),
    ("DLLL",  None,    "Dell 2x (GraniteShares)",                      "DELL"),
    ("NOWL",  None,    "ServiceNow 2x (GraniteShares)",                "NOW"),
    ("CRMG",  None,    "Salesforce 2x (Leverage Shares)",              "CRM"),
    ("ARMG",  None,    "ARM Holdings 2x (Leverage Shares)",            "ARM"),
    ("ADBG",  None,    "Adobe 2x (Leverage Shares)",                   "ADBE"),
    ("LLYX",  None,    "Eli Lilly 2x (Defiance)",                      "LLY"),
    ("PYPG",  None,    "PayPal 2x (Leverage Shares)",                  "PYPL"),
    ("INTW",  None,    "Intel 2x (GraniteShares)",                     "INTC"),
    ("ROBN",  "HOOG",  "Hood 2x (T-Rex / Leverage Shares)",           "HOOD"),
    ("SMCY",  None,    "YieldMax SMCI Strategy",                       "SMCI"),
    ("SQQU",  None,    "Block (SQ) 2x (GraniteShares)",               "SQ"),
    ("AFRX",  None,    "Affirm 2x (Defiance)",                         "AFRM"),
    ("QQQX",  None,    "Qualcomm 2x (Defiance)",                       "QCOM"),
]

# All leveraged pairs combined
LEVERAGED_ETF_PAIRS = INDEX_LEVERAGED + SECTOR_LEVERAGED + COMMODITY_LEVERAGED + SINGLE_STOCK_LEVERAGED

# ── Crypto ETFs (spot + leveraged) ──────────────────────────────────────
# These are NOT leveraged single-stock ETFs but standalone crypto products
CRYPTO_ETFS = [
    # Spot Bitcoin ETFs
    "IBIT",   # iShares Bitcoin Trust (BlackRock) — largest
    "FBTC",   # Fidelity Wise Origin Bitcoin
    "ARKB",   # ARK 21Shares Bitcoin ETF
    "BITB",   # Bitwise Bitcoin ETF
    "GBTC",   # Grayscale Bitcoin Trust (OTC converted)
    # Spot Ethereum ETFs
    "ETHA",   # iShares Ethereum Trust (BlackRock)
    "FETH",   # Fidelity Ethereum Fund
    # XRP spot ETF (2025)
    "XRPI",   # ProShares Ultra XRP (skipped automatically if not on yfinance)
    # Leveraged Bitcoin ETFs
    "BITU",   # ProShares Ultra Bitcoin 2x
    "BITI",   # ProShares Short Bitcoin (inverse hedge)
]

# ── S&P 500 stocks (top by market cap/liquidity, not already covered) ───
SP500_STOCKS = [
    # Energy majors
    "XOM", "CVX", "COP",
    # Consumer staples
    "WMT", "KO", "PEP", "PG", "COST", "MCD", "SBUX", "NKE", "DIS",
    # Financials
    "BAC", "GS", "MS", "BLK", "AXP", "WFC", "C",
    # Healthcare
    "JNJ", "ABBV", "AMGN", "BMY", "REGN", "GILD", "CVS", "MDT",
    # Industrials
    "GE", "CAT", "HON", "RTX", "BA", "UPS", "FDX", "DE",
    # Semiconductors (beyond what's covered)
    "QCOM", "TXN", "IBM", "AMAT", "LRCX", "KLAC", "ADI", "MRVL",
    # Data/analytics
    "SPGI", "MCO", "ICE", "CME",
    # Utilities/Telecom
    "NEE", "T", "VZ",
    # Real Estate
    "AMT", "PLD",
]

# ── Russell 2000 / Small-cap high-momentum stocks ───────────────────────
RUSSELL2000_STOCKS = [
    # Space & Advanced Air Mobility
    "ACHR", "JOBY", "LUNR", "RDW", "RKLB",
    # Crypto mining & blockchain
    "RIOT", "MARA", "HUT", "IREN", "CIFR", "CLBT",
    # AI / small-cap tech
    "AI", "BBAI", "RXRX", "DOCN", "IONQ",
    # Fintech
    "SQ", "AFRM", "UPST", "HOOD", "SOFI", "RDDT",
    # Biotech
    "SMAR", "BEAM", "EDIT", "NTLA",
    # High-momentum growth
    "CELH", "AXON", "CFLT", "DDOG", "MDB",
]

# Underlying market leaders (underlying assets for leveraged ETFs + standalone)
UNDERLYING_LEADERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    # Finance / consumer
    "JPM", "V", "MA", "UNH", "HD", "PG", "COST", "MRK",
    # Growth / software
    "AMD", "ORCL", "CRM", "NFLX", "PANW", "SNOW", "PLTR", "SHOP",
    # Crypto / alt
    "UBER", "COIN", "MSTR", "MARA",
    # Semi / hardware
    "MU", "SMCI", "IONQ", "TSM", "ARM", "ADBE",
    # Other
    "CRWD", "SOFI", "RKLB", "RDDT", "HOOD", "INTC", "NOW",
    "LLY", "PYPL", "DELL", "BABA",
]


def get_all_instruments() -> list[str]:
    """Return deduplicated, sorted list of all instruments."""
    instruments = set(INDEX_DRIVERS)
    for bull, bear, _, *_ in LEVERAGED_ETF_PAIRS:
        instruments.add(bull)
        if bear:
            instruments.add(bear)
    for stock in UNDERLYING_LEADERS:
        instruments.add(stock)
    for etf in CRYPTO_ETFS:
        instruments.add(etf)
    for stock in SP500_STOCKS:
        instruments.add(stock)
    for stock in RUSSELL2000_STOCKS:
        instruments.add(stock)
    return sorted(instruments)


ALL_INSTRUMENTS = get_all_instruments()

# ── Data Settings ────────────────────────────────────────────────────────

DATA_START_DATE = "2010-01-01"
DOWNLOAD_BATCH_SIZE = 20
DOWNLOAD_DELAY_SECONDS = 1.0

# ── Indicator Parameters ─────────────────────────────────────────────────

INDICATOR_PARAMS = {
    "rsi_period": 14,
    "atr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "sma_short": 50,
    "sma_long": 200,
}

# ── Probability Engine Parameters ────────────────────────────────────────

PROBABILITY_PARAMS = {
    "lookahead_days": 20,       # forward-looking window
    "target_return": 0.08,      # 8% return threshold for success label
}

# ── Signal Engine Parameters (3-Tier Trading) ───────────────────────────

SIGNAL_PARAMS = {
    "accumulate_atr_mult": 1.0,   # accumulate zone = price - 1 ATR
    "scale_atr_mult": 2.0,        # scale-out target = price + 2 ATR
    "sell_atr_mult": 4.0,         # full target = price + 4 ATR
}

# Confidence tier cutoffs (< 50% discarded)
TIER_THRESHOLDS = {
    "HIGH":   0.90,
    "MEDIUM": 0.70,
    "LOW":    0.50,
}

# ── Volatility Regime Parameters ─────────────────────────────────────────

VOLATILITY_PARAMS = {
    "percentile_lookback":       252,   # rolling window for percentile rank (1 trading year)
    "realized_vol_window":        20,   # 20-day realized volatility window
    "compression_bb_pct":       0.15,   # BB_width_percentile < 15% → compression
    "compression_atr_pct":      0.20,   # ATR_percentile < 20% → compression
    "expansion_volume_ratio":    1.5,   # volume_ratio > 1.5 → expansion candidate
    "expansion_momentum_bars":     3,   # momentum diff lookback for expansion
    "expansion_range_bars":        5,   # prior N bars used to define compression range
    "low_vol_composite_pct":    0.30,   # composite percentile < 30% → LOW_VOL
    "high_vol_composite_pct":   0.70,   # composite percentile > 70% → HIGH_VOL
}

# Volatility regime labels
VOL_LOW    = "LOW_VOL"
VOL_NORMAL = "NORMAL_VOL"
VOL_HIGH   = "HIGH_VOL"

# Primary leveraged ETF watchlist (spec-defined focus list)
PRIMARY_WATCHLIST = [
    "SOXL", "SOXS",    # Semiconductors 3x
    "TQQQ", "SQQQ",    # NASDAQ-100 3x
    "NVDL", "NVDS",    # NVIDIA 2x
    "TSLL", "TSLZ",    # Tesla 2x
    "MSTU", "MSTZ",    # MicroStrategy 2x
    "LABU", "LABD",    # Biotech 3x
    "FAS",  "FAZ",     # Financials 3x
    "GUSH", "DRIP",    # Oil & Gas 2x
    "BOIL", "KOLD",    # Natural Gas 2x
    "TNA",  "TZA",     # Russell 2000 3x
    "UPRO", "SPXU",    # S&P 500 3x
    "TECL", "TECS",    # Technology 3x
]
