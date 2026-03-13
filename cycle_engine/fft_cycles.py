"""Advanced FFT cycle detection with multi-resolution frequency analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import periodogram, detrend, welch


def detect_fft_cycles(
    close: np.ndarray,
    min_cycle: int = 5,
    max_cycle: int = 120,
    n_top: int = 3,
) -> list[dict]:
    """Detect dominant cycles using FFT with Welch's method for noise reduction.

    Returns up to n_top dominant cycles sorted by spectral power.
    """
    if len(close) < max_cycle:
        return []

    log_prices = np.log(close)
    detrended = detrend(log_prices, type="linear")

    # Welch's method: better noise rejection than raw periodogram
    nperseg = min(len(detrended), 256)
    freqs, power = welch(detrended, fs=1.0, nperseg=nperseg)

    min_freq = 1.0 / max_cycle
    max_freq = 1.0 / min_cycle
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not mask.any():
        return []

    valid_freqs = freqs[mask]
    valid_power = power[mask]
    total_power = float(valid_power.sum())
    if total_power <= 0:
        return []

    # Find top N peaks
    sorted_idx = np.argsort(valid_power)[::-1]
    cycles: list[dict] = []
    for i in range(min(n_top, len(sorted_idx))):
        idx = sorted_idx[i]
        freq = float(valid_freqs[idx])
        if freq <= 0:
            continue
        pw = float(valid_power[idx])
        cycles.append({
            "cycle_length": round(1.0 / freq, 2),
            "strength": round(pw / total_power, 4),
            "frequency": round(freq, 6),
        })

    return cycles


def dominant_fft_cycle(close: np.ndarray, min_cycle: int = 5, max_cycle: int = 120) -> tuple[float, float]:
    """Return (cycle_length, strength) for the single strongest FFT cycle."""
    cycles = detect_fft_cycles(close, min_cycle, max_cycle, n_top=1)
    if not cycles:
        return 0.0, 0.0
    return cycles[0]["cycle_length"], cycles[0]["strength"]
