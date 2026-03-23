"""
Trade Journal API
Logs every trade with full signal metadata, screenshots, and outcomes.
"""

import json
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter()

# Data directories
TRADES_DIR = Path("/Users/asmian/RoboAlgo/data/trades")
SCREENSHOTS_DIR = Path("/Users/asmian/RoboAlgo/data/screenshots")

# Ensure directories exist
TRADES_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Request Models ─────────────────────────────────────────────────────────────

class TradeEntryRequest(BaseModel):
    """Log a new trade entry"""
    symbol: str
    timeframe: str = "daily"
    entry_price: float
    target_price: float
    stop_price: float

    # All signal metadata
    signals: dict[str, Any]  # all_patterns, discord_gate, confluence_score, gex_state, etc.

    # Screenshot as base64
    screenshot_base64: Optional[str] = None

    notes: str = ""


class TradeOutcomeRequest(BaseModel):
    """Update trade outcome"""
    trade_id: str
    exit_price: float
    exit_time: Optional[str] = None  # ISO 8601, defaults to now
    reason: str = ""  # hit_target, stopped_out, manual_exit, etc.


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/log-entry")
def log_trade_entry(req: TradeEntryRequest) -> dict[str, str]:
    """
    Log a new trade entry with signals and screenshot.

    Returns: { trade_id, path }
    """
    trade_id = f"{req.symbol}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    entry_time = datetime.utcnow().isoformat() + "Z"

    # Create trade JSON object
    trade_data = {
        "trade_id": trade_id,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "entry_time": entry_time,
        "entry_price": req.entry_price,
        "target_price": req.target_price,
        "stop_price": req.stop_price,

        "signals": req.signals,

        "screenshot": None,  # Will be set if screenshot provided

        "outcome": {
            "status": "open",
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "pnl_percent": None,
            "bars_held": None,
            "reason": None
        },

        "notes": req.notes
    }

    # Save screenshot if provided
    if req.screenshot_base64:
        try:
            # Decode base64 image
            image_data = base64.b64decode(req.screenshot_base64)

            # Create date subdirectory
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            date_dir = SCREENSHOTS_DIR / date_str
            date_dir.mkdir(parents=True, exist_ok=True)

            # Save screenshot
            screenshot_filename = f"{trade_id}.png"
            screenshot_path = date_dir / screenshot_filename
            screenshot_path.write_bytes(image_data)

            # Store reference in trade data
            trade_data["screenshot"] = {
                "path": f"data/screenshots/{date_str}/{screenshot_filename}",
                "timestamp": entry_time,
                "size_bytes": len(image_data)
            }
        except Exception as e:
            # Log error but don't fail the trade entry
            print(f"Screenshot save failed for {trade_id}: {e}")

    # Save trade JSON
    trade_file = TRADES_DIR / f"{trade_id}.json"
    try:
        trade_file.write_text(json.dumps(trade_data, indent=2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save trade: {e}")

    return {
        "trade_id": trade_id,
        "path": str(trade_file)
    }


@router.patch("/log-outcome/{trade_id}")
def log_trade_outcome(trade_id: str, req: TradeOutcomeRequest) -> dict[str, Any]:
    """
    Update trade outcome when closed.

    Returns: updated trade data
    """
    # Find trade file
    trade_file = None
    for f in TRADES_DIR.glob("*.json"):
        if trade_id in f.name:
            trade_file = f
            break

    if not trade_file or not trade_file.exists():
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    # Load trade
    try:
        trade_data = json.loads(trade_file.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trade: {e}")

    # Update outcome
    exit_time = req.exit_time or datetime.utcnow().isoformat() + "Z"
    entry_price = trade_data["entry_price"]
    pnl = (req.exit_price - entry_price) * 100  # Assuming 100 shares for now
    pnl_percent = ((req.exit_price - entry_price) / entry_price) * 100

    trade_data["outcome"] = {
        "status": "closed",
        "exit_time": exit_time,
        "exit_price": req.exit_price,
        "pnl": round(pnl, 2),
        "pnl_percent": round(pnl_percent, 2),
        "bars_held": None,  # Could calculate if we had bar timestamps
        "reason": req.reason
    }

    # Save updated trade
    try:
        trade_file.write_text(json.dumps(trade_data, indent=2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save outcome: {e}")

    return trade_data


@router.get("/get/{trade_id}")
def get_trade(trade_id: str) -> dict[str, Any]:
    """Retrieve a specific trade."""
    trade_file = None
    for f in TRADES_DIR.glob("*.json"):
        if trade_id in f.name:
            trade_file = f
            break

    if not trade_file or not trade_file.exists():
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    try:
        return json.loads(trade_file.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trade: {e}")


@router.get("/list")
def list_trades(symbol: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
    """
    List all trades, optionally filtered by symbol or status.

    Query params:
      - symbol: filter by symbol (e.g. "AAPL")
      - status: filter by status (open, closed)
    """
    trades = []

    for trade_file in sorted(TRADES_DIR.glob("*.json"), reverse=True):
        try:
            trade_data = json.loads(trade_file.read_text())

            # Apply filters
            if symbol and trade_data["symbol"] != symbol.upper():
                continue
            if status and trade_data["outcome"]["status"] != status:
                continue

            trades.append(trade_data)
        except Exception as e:
            print(f"Failed to read {trade_file}: {e}")
            continue

    return trades


@router.get("/stats")
def get_stats(symbol: Optional[str] = None) -> dict[str, Any]:
    """
    Calculate stats on logged trades.

    Returns: win_rate, avg_win, avg_loss, profit_factor, etc.
    """
    trades = list_trades(symbol=symbol)

    closed_trades = [t for t in trades if t["outcome"]["status"] == "closed"]
    if not closed_trades:
        return {"error": "No closed trades found"}

    wins = [t for t in closed_trades if t["outcome"]["pnl"] > 0]
    losses = [t for t in closed_trades if t["outcome"]["pnl"] < 0]

    total_pnl = sum(t["outcome"]["pnl"] for t in closed_trades)
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0

    avg_win = sum(t["outcome"]["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(abs(t["outcome"]["pnl"]) for t in losses) / len(losses) if losses else 0

    profit_factor = avg_win / avg_loss if avg_loss > 0 else (1.0 if avg_win > 0 else 0.0)

    return {
        "total_trades": len(closed_trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_percent": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2)
    }
