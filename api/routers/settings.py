"""
api/routers/settings.py
────────────────────────────────────────────────────────────────────────────────
Data-provider settings — read, write, and test API keys at runtime.

Endpoints:
  GET  /api/settings              Return current settings (keys masked)
  POST /api/settings              Save new settings to settings.json + apply to runtime
  POST /api/settings/test/{provider}  Live-test a provider key
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Settings file path ────────────────────────────────────────────────────────
_SETTINGS_FILE = Path(__file__).resolve().parents[2] / "settings.json"

_DEFAULT_SETTINGS: dict[str, Any] = {
    "alphavantage": {
        "api_key": "",
        "rpm":     5,
        "tier":    "free",
        "enabled": True,
    },
    "finnhub": {
        "api_key": "",
        "enabled": True,
    },
    "alpaca": {
        "api_key":    "",
        "api_secret": "",
        "base_url":   "https://data.alpaca.markets",
        "enabled":    False,
    },
    "data_sources": {
        "intraday_mtf": "yfinance",
        "daily_ohlcv":  "alphavantage",
        "options":      "yahoo",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load() -> dict[str, Any]:
    """Load settings.json; fall back to defaults if missing or corrupt."""
    if not _SETTINGS_FILE.exists():
        return dict(_DEFAULT_SETTINGS)
    try:
        with _SETTINGS_FILE.open() as f:
            raw = json.load(f)
        # Merge with defaults so new keys always exist
        merged = dict(_DEFAULT_SETTINGS)
        for section, vals in raw.items():
            if section in merged and isinstance(merged[section], dict):
                merged[section] = {**merged[section], **vals}
            else:
                merged[section] = vals
        return merged
    except Exception as exc:
        logger.warning("settings.json unreadable (%s) — using defaults", exc)
        return dict(_DEFAULT_SETTINGS)


def _save(settings: dict[str, Any]) -> None:
    """Write settings.json atomically."""
    tmp = _SETTINGS_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(settings, f, indent=2)
    tmp.replace(_SETTINGS_FILE)


def _mask(key: str) -> str:
    """Show only last 4 chars of a secret key."""
    if not key or len(key) < 5:
        return "••••" if key else ""
    return "•" * (len(key) - 4) + key[-4:]


def _apply_to_runtime(settings: dict[str, Any]) -> None:
    """
    Push saved settings into os.environ and reload affected module globals.
    This makes changes take effect immediately without a backend restart
    for most providers.
    """
    av = settings.get("alphavantage", {})
    if av.get("api_key"):
        os.environ["ALPHAVANTAGE_API_KEY"] = av["api_key"]
    if av.get("rpm"):
        os.environ["ALPHAVANTAGE_RPM"] = str(av["rpm"])

    fh = settings.get("finnhub", {})
    if fh.get("api_key"):
        os.environ["FINNHUB_API_KEY"] = fh["api_key"]

    alp = settings.get("alpaca", {})
    if alp.get("api_key"):
        os.environ["APCA_API_KEY_ID"] = alp["api_key"]
    if alp.get("api_secret"):
        os.environ["APCA_API_SECRET_KEY"] = alp["api_secret"]

    # Reload alphavantage_fetcher module-level constants live
    try:
        import data_engine.alphavantage_fetcher as av_mod
        av_mod._AV_KEY       = av.get("api_key") or av_mod._AV_KEY
        av_mod._AV_RPM       = int(av.get("rpm", av_mod._AV_RPM))
        av_mod._REQUEST_DELAY = max(0.10, 60.0 / max(av_mod._AV_RPM, 1))
        av_mod._IS_PREMIUM   = av_mod._AV_RPM >= 30
        av_mod._MAX_WORKERS  = 8 if av_mod._IS_PREMIUM else 4
        # Bust intraday cache so next scan uses new key
        av_mod._intraday_cache.clear()
        logger.info(
            "AV config reloaded: rpm=%d premium=%s", av_mod._AV_RPM, av_mod._IS_PREMIUM
        )
    except Exception as exc:
        logger.warning("Could not reload AV module: %s", exc)


def _tier_label(rpm: int) -> str:
    if rpm >= 600:
        return "professional"
    if rpm >= 30:
        return "premium"
    return "free"


# ── Pydantic models ───────────────────────────────────────────────────────────

class AvSettings(BaseModel):
    api_key: str = ""
    rpm:     int = 5
    enabled: bool = True


class FinnhubSettings(BaseModel):
    api_key: str = ""
    enabled: bool = True


class AlpacaSettings(BaseModel):
    api_key:    str = ""
    api_secret: str = ""
    base_url:   str = "https://data.alpaca.markets"
    enabled:    bool = False


class DataSourceSettings(BaseModel):
    intraday_mtf: str = "yfinance"   # "alphavantage" | "yfinance"
    daily_ohlcv:  str = "alphavantage"
    options:      str = "yahoo"


class SettingsPayload(BaseModel):
    alphavantage:  AvSettings          = AvSettings()
    finnhub:       FinnhubSettings     = FinnhubSettings()
    alpaca:        AlpacaSettings      = AlpacaSettings()
    data_sources:  DataSourceSettings  = DataSourceSettings()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def get_settings():
    """Return current settings with API keys masked for display."""
    s = _load()
    av  = s.get("alphavantage", {})
    fh  = s.get("finnhub", {})
    alp = s.get("alpaca", {})

    return {
        "alphavantage": {
            "api_key_masked": _mask(av.get("api_key", "")),
            "api_key_set":    bool(av.get("api_key")),
            "rpm":            av.get("rpm", 5),
            "tier":           _tier_label(av.get("rpm", 5)),
            "enabled":        av.get("enabled", True),
        },
        "finnhub": {
            "api_key_masked": _mask(fh.get("api_key", "")),
            "api_key_set":    bool(fh.get("api_key")),
            "enabled":        fh.get("enabled", True),
        },
        "alpaca": {
            "api_key_masked":    _mask(alp.get("api_key", "")),
            "api_secret_masked": _mask(alp.get("api_secret", "")),
            "api_key_set":       bool(alp.get("api_key")),
            "api_secret_set":    bool(alp.get("api_secret")),
            "base_url":          alp.get("base_url", "https://data.alpaca.markets"),
            "enabled":           alp.get("enabled", False),
        },
        "data_sources": s.get("data_sources", _DEFAULT_SETTINGS["data_sources"]),
    }


@router.post("")
def save_settings(payload: SettingsPayload):
    """
    Save new settings.  Merges with existing file so keys not sent are preserved.
    Applies changes to the running process immediately (no restart needed for most).
    """
    current = _load()

    # Only overwrite keys that are actually provided (non-empty strings keep old values)
    av = current.setdefault("alphavantage", {})
    if payload.alphavantage.api_key:
        av["api_key"] = payload.alphavantage.api_key
    av["rpm"]     = payload.alphavantage.rpm
    av["tier"]    = _tier_label(payload.alphavantage.rpm)
    av["enabled"] = payload.alphavantage.enabled

    fh = current.setdefault("finnhub", {})
    if payload.finnhub.api_key:
        fh["api_key"] = payload.finnhub.api_key
    fh["enabled"] = payload.finnhub.enabled

    alp = current.setdefault("alpaca", {})
    if payload.alpaca.api_key:
        alp["api_key"] = payload.alpaca.api_key
    if payload.alpaca.api_secret:
        alp["api_secret"] = payload.alpaca.api_secret
    alp["base_url"] = payload.alpaca.base_url
    alp["enabled"]  = payload.alpaca.enabled

    current["data_sources"] = payload.data_sources.model_dump()

    _save(current)
    _apply_to_runtime(current)

    logger.info("Settings saved and applied")
    return {"ok": True, "message": "Settings saved and applied to running process."}


@router.post("/test/{provider}")
def test_provider(provider: str, payload: dict = None):
    """
    Live-test a provider API key.  Accepts optional {api_key, api_secret} body.
    Falls back to saved key if not provided.
    """
    if payload is None:
        payload = {}
    current = _load()
    start = time.perf_counter()

    if provider == "alphavantage":
        key = payload.get("api_key") or current.get("alphavantage", {}).get("api_key", "")
        if not key:
            raise HTTPException(400, "No Alpha Vantage key configured")
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            elapsed = round((time.perf_counter() - start) * 1000)
            if "Note" in data:
                return {"ok": False, "error": "Rate limited — key may be free tier", "elapsed_ms": elapsed}
            if "Information" in data:
                return {"ok": False, "error": data["Information"][:120], "elapsed_ms": elapsed}
            gq = data.get("Global Quote", {})
            price = gq.get("05. price", "?")
            rpm = current.get("alphavantage", {}).get("rpm", 5)
            return {
                "ok":        True,
                "provider":  "alphavantage",
                "tier":      _tier_label(rpm),
                "rpm":       rpm,
                "sample":    f"AAPL @ ${price}",
                "elapsed_ms": elapsed,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "elapsed_ms": round((time.perf_counter() - start) * 1000)}

    elif provider == "finnhub":
        key = payload.get("api_key") or current.get("finnhub", {}).get("api_key", "")
        if not key:
            raise HTTPException(400, "No Finnhub key configured")
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            elapsed = round((time.perf_counter() - start) * 1000)
            if "error" in data:
                return {"ok": False, "error": data["error"], "elapsed_ms": elapsed}
            price = data.get("c", "?")
            return {"ok": True, "provider": "finnhub", "sample": f"AAPL @ ${price}", "elapsed_ms": elapsed}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "elapsed_ms": round((time.perf_counter() - start) * 1000)}

    elif provider == "alpaca":
        key    = payload.get("api_key")    or current.get("alpaca", {}).get("api_key", "")
        secret = payload.get("api_secret") or current.get("alpaca", {}).get("api_secret", "")
        base   = current.get("alpaca", {}).get("base_url", "https://data.alpaca.markets")
        if not key or not secret:
            raise HTTPException(400, "Alpaca requires both api_key and api_secret")
        url = f"{base}/v2/stocks/bars?symbols=AAPL&timeframe=1Day&limit=1"
        try:
            r = requests.get(url, headers={
                "APCA-API-KEY-ID":     key,
                "APCA-API-SECRET-KEY": secret,
            }, timeout=10)
            elapsed = round((time.perf_counter() - start) * 1000)
            if r.status_code == 403:
                return {"ok": False, "error": "403 Forbidden — check key/secret or subscription tier", "elapsed_ms": elapsed}
            r.raise_for_status()
            data = r.json()
            bars = data.get("bars", {}).get("AAPL", [])
            sample = f"AAPL close=${bars[-1]['c']}" if bars else "Connected (no bars)"
            return {"ok": True, "provider": "alpaca", "sample": sample, "elapsed_ms": elapsed}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "elapsed_ms": round((time.perf_counter() - start) * 1000)}

    elif provider == "yfinance":
        try:
            import yfinance as yf
            hist = yf.Ticker("AAPL").history(period="2d", interval="1d")
            elapsed = round((time.perf_counter() - start) * 1000)
            if hist.empty:
                return {"ok": False, "error": "No data returned", "elapsed_ms": elapsed}
            price = round(float(hist["Close"].iloc[-1]), 2)
            return {"ok": True, "provider": "yfinance", "tier": "free", "sample": f"AAPL @ ${price}", "elapsed_ms": elapsed}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "elapsed_ms": round((time.perf_counter() - start) * 1000)}

    else:
        raise HTTPException(404, f"Unknown provider: {provider}")
