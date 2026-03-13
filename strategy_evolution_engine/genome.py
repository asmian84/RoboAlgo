"""Strategy genome: parameter set defining a complete trading strategy."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StrategyGenomeParams:
    """A single strategy genome — a set of parameters that define entry/exit rules."""

    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Entry conditions
    entry_confluence_min: float = 60.0        # min confluence score to enter (0-100)
    pattern_type: str = "any"                  # pattern filter: "any", "compression_breakout", etc.
    regime_filter: str = "ALL"                 # COMPRESSION | TREND | EXPANSION | CHAOS | ALL

    # Position management
    stop_atr_mult: float = 2.0                # stop distance in ATR multiples
    target_atr_mult: float = 4.0              # target distance in ATR multiples
    hold_days_max: int = 20                    # max holding period

    # Risk management
    position_size_pct: float = 2.0            # % of portfolio per trade
    max_positions: int = 5                     # max concurrent positions

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyGenomeParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "StrategyGenomeParams":
        return cls.from_dict(json.loads(s))
