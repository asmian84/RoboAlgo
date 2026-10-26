"""Education Engine — tooltips, signal explanations, beginner/advanced mode."""
from .tooltip_system import TOOLTIPS, get_tooltip
from .signal_explainer import SignalExplainer

__all__ = ["TOOLTIPS", "get_tooltip", "SignalExplainer"]
