"""Wave Phase Filter — classifies price action as IMPULSE or CORRECTION.

IMPULSE   — strong directional movement (ATR expanding + trending structure).
            Reversal trades are high-risk; favour trend-continuation setups.

CORRECTION — pullback/consolidation (ATR contracting + swing compression).
             Optimal window for reversal and mean-reversion setups.

TRANSITION — mixed signals; low-confidence classification.

Output format::

    {
        "wave_phase":      "IMPULSE" | "CORRECTION" | "TRANSITION",
        "direction":       "UP" | "DOWN" | "NEUTRAL",
        "confidence":      float,        # 0–1
        "atr_state":       str,          # "EXPANDING" | "NORMAL" | "CONTRACTING"
        "amplitude_ratio": float,        # last swing range / prior swing range
        "swing_count":     int,
    }
"""

from wave_phase.wave_phase_engine import WavePhaseEngine, detect_wave_phase

__all__ = ["WavePhaseEngine", "detect_wave_phase"]
