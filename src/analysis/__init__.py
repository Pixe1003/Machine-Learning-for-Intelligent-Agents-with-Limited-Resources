"""Structural analysis: heat-maps, posterior-scatter, class gaps, fitting."""

from src.analysis.diagnostics import monitor_diagnostics, posterior_monotonicity_stats, transition_entropy_stats

__all__ = ["monitor_diagnostics", "posterior_monotonicity_stats", "transition_entropy_stats"]
