"""Core differentiable recursion for f(n, k)."""
from src.core.recursion import compute_error, state_distribution
from src.core.loss import (
    evaluate,
    evaluate_hard,
    evaluate_soft,
    majority_loss,
    report_evaluate,
)

__all__ = [
    "compute_error",
    "state_distribution",
    "majority_loss",
    "evaluate",
    "evaluate_soft",
    "evaluate_hard",
    "report_evaluate",
]
