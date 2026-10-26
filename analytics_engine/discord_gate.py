"""
analytics_engine/discord_gate.py
─────────────────────────────────────────────────────────────────────────────
Matrix Profile Discord Gate — invisible signal quality filter.

Uses STUMPY to compute the Matrix Profile of the recent price series and
identify "discord" bars: bars whose price action has no historical precedent
(nearest-neighbour distance is unusually large).

Signals fired during anomalous price action are suppressed — the market is
doing something structurally unusual and pattern-based rules lose their edge.

Usage (called by pattern_engine/service.py before emitting signals):

    from analytics_engine.discord_gate import is_anomalous

    if is_anomalous(close_series, window=20):
        return []   # suppress all signals

Design goals:
  • Fast: uses scrump (approximate Matrix Profile) for speed
  • Conservative: only suppresses on *clear* anomalies (top 5% discord)
  • Invisible: the user never sees this — signals simply don't appear
  • Fallback-safe: if STUMPY is unavailable, always returns False (no suppression)
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Discord suppression threshold — bars in the top N% most anomalous
# are considered structural anomalies.  Conservative: 5%.
_DISCORD_PCT_THRESHOLD = 0.05

# Minimum bars of history needed for a meaningful Matrix Profile
_MIN_BARS = 60

# Window size for Matrix Profile (bars) — captures 20-bar swing structures
_DEFAULT_WINDOW = 20


def is_anomalous(
    close: np.ndarray | list,
    window: int = _DEFAULT_WINDOW,
    pct_threshold: float = _DISCORD_PCT_THRESHOLD,
) -> bool:
    """
    Return True if the most recent bar is a Matrix Profile discord
    (price action with no historical precedent).

    Parameters
    ----------
    close : array-like
        Close prices, most recent last.
    window : int
        Matrix Profile sub-sequence length (bars).  Shorter = more sensitive.
    pct_threshold : float
        Top-N% most anomalous bars trigger suppression.  Default 5%.

    Returns
    -------
    bool
        True  → current price action is anomalous; suppress pattern signals.
        False → normal market behaviour; signals can fire.
    """
    try:
        import stumpy  # noqa: PLC0415
    except ImportError:
        logger.debug("STUMPY not available — discord gate disabled")
        return False

    arr = np.asarray(close, dtype=float)
    n   = len(arr)

    if n < _MIN_BARS or window >= n // 2:
        return False

    # Remove NaN/Inf
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if len(arr) < _MIN_BARS:
            return False

    try:
        # scrump is approximate but fast — sufficient for a gate check.
        # percentage=0.02 → 2% of the full Matrix Profile computed per call;
        # we only need the profile value at the last position, so a rough
        # approximation is fine.
        mp = stumpy.stump(arr, m=window)
        profile = mp[:, 0].astype(float)   # Matrix Profile distances

        if len(profile) < 2:
            return False

        # The last bar's profile value (how unusual is the current window?)
        last_mp = float(profile[-1])

        # Discord threshold: value that only pct_threshold of bars exceed
        threshold = float(np.percentile(profile, (1 - pct_threshold) * 100))

        is_discord = last_mp > threshold
        if is_discord:
            logger.info(
                "Discord gate: last bar MP=%.3f > threshold=%.3f (top %.0f%%) — suppressing signals",
                last_mp, threshold, pct_threshold * 100,
            )
        return is_discord

    except Exception as exc:
        logger.debug("Discord gate computation failed: %s", exc)
        return False


def discord_score(
    close: np.ndarray | list,
    window: int = _DEFAULT_WINDOW,
) -> float:
    """
    Return the current bar's discord percentile (0.0–1.0).

    0.0 = perfectly ordinary price action (low anomaly).
    1.0 = completely unprecedented price action (extreme discord).

    Used by the Command Center market safety panel to show an "anomaly level"
    gauge without blocking signals — a softer signal than the binary gate.
    """
    try:
        import stumpy  # noqa: PLC0415
    except ImportError:
        return 0.0

    arr = np.asarray(close, dtype=float)
    n   = len(arr)

    if n < _MIN_BARS or window >= n // 2:
        return 0.0

    arr = arr[np.isfinite(arr)]
    if len(arr) < _MIN_BARS:
        return 0.0

    try:
        mp      = stumpy.stump(arr, m=window)
        profile = mp[:, 0].astype(float)
        last_mp = float(profile[-1])
        # Percentile rank of the last bar
        rank    = float(np.mean(profile <= last_mp))
        return round(rank, 3)
    except Exception:
        return 0.0
