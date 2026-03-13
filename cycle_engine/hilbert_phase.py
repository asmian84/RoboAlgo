"""Hilbert Transform phase analysis for instantaneous cycle measurement."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, detrend, butter, filtfilt


def compute_hilbert_phase(
    close: np.ndarray,
    dominant_cycle: float = 20.0,
    bandwidth: float = 0.3,
) -> dict:
    """Compute instantaneous phase and amplitude via Hilbert Transform.

    Uses a bandpass filter centered on the dominant cycle to isolate
    the cycle component before applying the Hilbert transform.

    Returns dict with:
        phase: float (0-1 normalized)
        amplitude: float (current cycle amplitude)
        phase_velocity: float (rate of phase change — speed through cycle)
    """
    n = len(close)
    if n < 60 or dominant_cycle < 3:
        return {"phase": 0.0, "amplitude": 0.0, "phase_velocity": 0.0}

    log_prices = np.log(close)
    detrended = detrend(log_prices, type="linear")

    # Bandpass filter around the dominant cycle
    center_freq = 1.0 / dominant_cycle
    low_freq = center_freq * (1.0 - bandwidth)
    high_freq = center_freq * (1.0 + bandwidth)
    nyquist = 0.5  # sampling rate = 1 bar/day

    low_freq = max(low_freq, 0.001)
    high_freq = min(high_freq, nyquist * 0.95)

    if low_freq >= high_freq:
        return {"phase": 0.0, "amplitude": 0.0, "phase_velocity": 0.0}

    try:
        b, a = butter(2, [low_freq / nyquist, high_freq / nyquist], btype="band")
        filtered = filtfilt(b, a, detrended)
    except Exception:
        filtered = detrended

    # Hilbert transform
    analytic_signal = hilbert(filtered)
    inst_phase = np.angle(analytic_signal)
    inst_amplitude = np.abs(analytic_signal)

    # Current values
    current_phase = float(inst_phase[-1])
    current_amplitude = float(inst_amplitude[-1])

    # Normalize phase to 0-1
    normalized_phase = (current_phase + np.pi) / (2 * np.pi)

    # Phase velocity (rate of phase change over last 5 bars)
    if n >= 6:
        phase_diff = np.diff(np.unwrap(inst_phase[-6:]))
        phase_velocity = float(np.mean(phase_diff))
    else:
        phase_velocity = 0.0

    return {
        "phase": round(float(normalized_phase), 4),
        "amplitude": round(current_amplitude, 6),
        "phase_velocity": round(phase_velocity, 6),
    }
