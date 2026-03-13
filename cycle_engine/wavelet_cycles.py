"""Wavelet-based cycle detection using continuous wavelet transform.

Note: scipy.signal.cwt was removed in scipy 1.12+.  We implement the CWT
manually by convolving the signal with scaled Morlet wavelets generated via
scipy.signal.morlet2 (which is still present).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend


def _morlet2(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """Pure-NumPy replacement for scipy.signal.morlet2 (removed in scipy 1.12+).

    Generates a complex Morlet wavelet of length M at scale s with angular
    frequency w.  Matches the normalisation used by the removed scipy function:
        ψ(t) = (π·s²)^(-1/4) · exp(i·w·t/s) · exp(-t²/(2·s²))
    """
    x = np.arange(0, M) - (M - 1.0) / 2.0   # centred sample indices
    x_s = x / s
    norm = (np.pi * s ** 2) ** (-0.25)
    return norm * np.exp(1j * w * x_s) * np.exp(-0.5 * x_s ** 2)


def _manual_cwt(signal: np.ndarray, scales: np.ndarray, w: float = 5.0) -> np.ndarray:
    """Compute a CWT power matrix [n_scales × n_samples].

    For each scale s we build a Morlet kernel whose length is 10*s samples
    (capped at len(signal)), convolve with the signal, and return the squared
    magnitude (power).
    """
    n = len(signal)
    n_scales = len(scales)
    power = np.zeros((n_scales, n), dtype=np.float64)

    for i, s in enumerate(scales):
        M = min(int(10 * s) + 1, n)
        if M % 2 == 0:
            M += 1
        kernel = _morlet2(M, s=s, w=w)
        conv = np.convolve(signal.astype(np.float64), np.conj(kernel[::-1]), mode="same")
        power[i] = np.abs(conv) ** 2

    return power


def detect_wavelet_cycles(
    close: np.ndarray,
    min_cycle: int = 5,
    max_cycle: int = 120,
) -> tuple[float, float]:
    """Detect dominant cycle using continuous wavelet transform (Morlet wavelet).

    Returns (cycle_length, strength) of the dominant cycle.
    """
    if len(close) < max_cycle:
        return 0.0, 0.0

    log_prices = np.log(close)
    detrended = detrend(log_prices, type="linear")

    # Scale range: each scale maps to an approximate period
    # For Morlet wavelet with w=5: period ≈ scale * 2π / w
    w = 5.0
    scales = np.arange(min_cycle, max_cycle + 1, dtype=float)

    try:
        power = _manual_cwt(detrended, scales, w=w)
    except Exception:
        return 0.0, 0.0

    # Average power across time for each scale
    avg_power = power.mean(axis=1)

    if avg_power.sum() <= 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(avg_power))
    cycle_length = float(scales[peak_idx])
    strength = float(avg_power[peak_idx] / avg_power.sum())

    return round(cycle_length, 2), round(strength, 4)
