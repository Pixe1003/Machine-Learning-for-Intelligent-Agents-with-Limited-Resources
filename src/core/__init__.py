"""Core differentiable recursion for f(n, k)."""
from src.core.recursion import compute_error, state_distribution
from src.core.loss import majority_loss, report_evaluate

__all__ = ["compute_error", "state_distribution", "majority_loss", "report_evaluate"]
