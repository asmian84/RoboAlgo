"""Cycle projection engine: forecasts next peak/trough dates and prices."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cycle_engine.fft_cycles import dominant_fft_cycle
from cycle_engine.wavelet_cycles import detect_wavelet_cycles
from cycle_engine.hilbert_phase import compute_hilbert_phase


def project_cycle(
    df: pd.DataFrame,
    min_cycle: int = 5,
    max_cycle: int = 120,
) -> dict[str, Any]:
    """Full cycle analysis and projection for a price series.

    Combines FFT, wavelet, and Hilbert methods to produce a consensus
    dominant cycle and project next peak/trough.

    Args:
        df: DataFrame with 'date' and 'close' columns.

    Returns:
        Dict with cycle parameters, phase, and projections.
    """
    result: dict[str, Any] = {
        "dominant_cycle_length": 0.0,
        "cycle_strength": 0.0,
        "cycle_phase": 0.0,
        "fft_cycle_length": 0.0,
        "fft_strength": 0.0,
        "wavelet_cycle_length": 0.0,
        "wavelet_strength": 0.0,
        "hilbert_phase": 0.0,
        "hilbert_amplitude": 0.0,
        "next_peak_date": None,
        "next_trough_date": None,
        "next_peak_price": None,
        "next_trough_price": None,
        "cycle_alignment_score": 0.0,
    }

    if df is None or len(df) < max_cycle:
        return result

    close = df["close"].astype(float).values
    dates = pd.to_datetime(df["date"]).values

    # ── FFT analysis ──────────────────────────────────────────────────────
    fft_length, fft_strength = dominant_fft_cycle(close, min_cycle, max_cycle)
    result["fft_cycle_length"] = fft_length
    result["fft_strength"] = fft_strength

    # ── Wavelet analysis ──────────────────────────────────────────────────
    wav_length, wav_strength = detect_wavelet_cycles(close, min_cycle, max_cycle)
    result["wavelet_cycle_length"] = wav_length
    result["wavelet_strength"] = wav_strength

    # ── Consensus dominant cycle (weighted average) ────────────────────────
    total_weight = fft_strength + wav_strength
    if total_weight > 0 and (fft_length > 0 or wav_length > 0):
        # Weight by strength
        if fft_length > 0 and wav_length > 0:
            dominant = (fft_length * fft_strength + wav_length * wav_strength) / total_weight
        elif fft_length > 0:
            dominant = fft_length
        else:
            dominant = wav_length
        strength = max(fft_strength, wav_strength)
    else:
        dominant = 0.0
        strength = 0.0

    result["dominant_cycle_length"] = round(dominant, 2)
    result["cycle_strength"] = round(strength, 4)

    if dominant < 3:
        return result

    # ── Hilbert phase ─────────────────────────────────────────────────────
    hilbert = compute_hilbert_phase(close, dominant_cycle=dominant)
    result["hilbert_phase"] = hilbert["phase"]
    result["hilbert_amplitude"] = hilbert["amplitude"]
    result["cycle_phase"] = hilbert["phase"]

    # ── Project next peak/trough ───────────────────────────────────────────
    last_date = pd.Timestamp(dates[-1])
    phase = hilbert["phase"]  # 0-1

    # Peak at phase 0.25, trough at phase 0.75 (sine cycle convention)
    # Bars to next peak: (0.25 - phase) * cycle_length (mod cycle_length)
    bars_to_peak = ((0.25 - phase) % 1.0) * dominant
    bars_to_trough = ((0.75 - phase) % 1.0) * dominant

    next_peak = last_date + timedelta(days=int(round(bars_to_peak * 1.4)))  # ~1.4 cal days per trading day
    next_trough = last_date + timedelta(days=int(round(bars_to_trough * 1.4)))

    result["next_peak_date"] = next_peak.date()
    result["next_trough_date"] = next_trough.date()

    # Estimate peak/trough prices from current price + amplitude
    current_price = float(close[-1])
    amp = hilbert["amplitude"]
    # Amplitude is in log-price space; convert to price space
    price_amp = current_price * (np.exp(amp) - 1.0) if amp > 0 else 0.0
    result["next_peak_price"] = round(current_price + price_amp, 4)
    result["next_trough_price"] = round(current_price - price_amp, 4)

    # ── Cycle alignment score (0-100) ──────────────────────────────────────
    # High score when: strong cycle + phase near trough (buying opportunity)
    # or strong cycle + phase near peak (selling signal)
    phase_proximity = min(abs(phase - 0.75), abs(phase - 0.25))  # distance to nearest peak/trough
    phase_score = max(0, 1.0 - phase_proximity * 4)  # 1.0 at peak/trough, 0 at midpoint
    alignment = strength * 50 + phase_score * 50
    result["cycle_alignment_score"] = round(float(np.clip(alignment, 0, 100)), 2)

    return result
