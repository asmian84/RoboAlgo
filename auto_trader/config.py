"""
auto_trader/config.py
────────────────────────────────────────────────────────────────────────────────
Persisted settings for the auto-trader loop, plus the live-mode safety gate.

Mirrors the load/merge/atomic-save pattern used by api/routers/settings.py:
  _load()  merges stored values onto dataclass defaults
  _save()  writes a `.tmp` file then atomically replaces the config file

Live trading requires BOTH:
  - the ROBOALGO_ALLOW_LIVE=1 environment variable, AND
  - cfg.live_confirmed is True

`save()` refuses to ever persist a config that requests live mode unless
ROBOALGO_ALLOW_LIVE=1 is already set in the environment — this prevents a UI
bug or bad request body from silently arming live trading.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from auto_trader.governor import RiskLimits

logger = logging.getLogger(__name__)

# ── Config file path ─────────────────────────────────────────────────────────
_CONFIG_FILE = Path(__file__).resolve().parents[1] / "auto_trader_config.json"


class AutoTraderConfigError(RuntimeError):
    """Raised when a config value or transition would violate the live-mode gate."""


@dataclass(frozen=True)
class AutoTraderConfig:
    enabled: bool = False              # master on/off
    mode: str = "paper"                # "paper" | "live"
    live_confirmed: bool = False       # half of the live gate
    broker: str = "paper"
    asset_class: str = "equity"        # "equity" | "crypto"
    profile: str = "SWING"
    symbols: list[str] = field(default_factory=list)  # empty -> scanner default universe
    max_symbols: int = 30
    max_workers: int = 2
    max_new_positions_per_cycle: int = 2
    ignore_market_hours: bool = False  # crypto only
    dry_run: bool = True               # SHIPS SAFE - default True
    risk: RiskLimits = field(default_factory=RiskLimits)


_DEFAULTS = AutoTraderConfig()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load() -> AutoTraderConfig:
    """Load auto_trader_config.json; merge stored values onto defaults."""
    if not _CONFIG_FILE.exists():
        return _DEFAULTS
    try:
        with _CONFIG_FILE.open() as f:
            raw = json.load(f)
        merged = asdict(_DEFAULTS)
        for key, val in raw.items():
            if key == "risk" and isinstance(val, dict):
                merged["risk"] = {**merged["risk"], **val}
            else:
                merged[key] = val
        merged["risk"] = RiskLimits(**merged["risk"])
        return AutoTraderConfig(**merged)
    except Exception as exc:
        logger.warning("auto_trader_config.json unreadable (%s) — using defaults", exc)
        return _DEFAULTS


def _save(cfg: AutoTraderConfig) -> None:
    """Write auto_trader_config.json atomically."""
    tmp = _CONFIG_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(as_dict(cfg), f, indent=2)
    tmp.replace(_CONFIG_FILE)


def as_dict(cfg: AutoTraderConfig) -> dict[str, Any]:
    """Serialize an AutoTraderConfig (including nested RiskLimits) to a plain dict."""
    return asdict(cfg)


def load() -> AutoTraderConfig:
    """Return the current auto-trader config, merged onto safe defaults."""
    return _load()


def save(partial: dict[str, Any]) -> AutoTraderConfig:
    """
    Merge `partial` onto the current config and persist it.

    Refuses to persist mode="live" or live_confirmed=True unless
    ROBOALGO_ALLOW_LIVE=1 is already set in the environment.
    """
    allow_live = os.getenv("ROBOALGO_ALLOW_LIVE") == "1"

    if not allow_live:
        if partial.get("mode") == "live":
            raise AutoTraderConfigError(
                "Refusing to save mode='live': ROBOALGO_ALLOW_LIVE=1 is not set."
            )
        if partial.get("live_confirmed") is True:
            raise AutoTraderConfigError(
                "Refusing to save live_confirmed=True: ROBOALGO_ALLOW_LIVE=1 is not set."
            )

    current = _load()
    merged = asdict(current)
    for key, val in partial.items():
        if key == "risk" and isinstance(val, dict):
            merged["risk"] = {**merged["risk"], **val}
        else:
            merged[key] = val
    merged["risk"] = RiskLimits(**merged["risk"]) if isinstance(merged["risk"], dict) else merged["risk"]

    new_cfg = replace(current, **{k: v for k, v in merged.items() if k != "risk"}, risk=merged["risk"])
    _save(new_cfg)
    return new_cfg


def assert_mode_safe(broker: Any, cfg: AutoTraderConfig) -> None:
    """
    Raise AutoTraderConfigError unless the config and broker agree it is safe
    to trade in the configured mode.

    Live trading requires BOTH the ROBOALGO_ALLOW_LIVE=1 env var AND
    cfg.live_confirmed is True — this holds regardless of what the broker
    reports, and regardless of what cfg.mode says.
    """
    allow_live = os.getenv("ROBOALGO_ALLOW_LIVE") == "1"
    live_gate_open = allow_live and cfg.live_confirmed is True

    if cfg.mode == "live" and not live_gate_open:
        raise AutoTraderConfigError(
            "cfg.mode='live' but the live gate is not open "
            "(requires ROBOALGO_ALLOW_LIVE=1 and live_confirmed=True)."
        )

    if broker.is_paper is False and not live_gate_open:
        raise AutoTraderConfigError(
            f"Broker '{getattr(broker, 'name', broker)}' reports live trading, "
            "but the live gate is not open "
            "(requires ROBOALGO_ALLOW_LIVE=1 and live_confirmed=True)."
        )

    if cfg.mode == "paper" and broker.is_paper is False:
        raise AutoTraderConfigError(
            "config says paper, broker reports live"
        )
