"""
Nesting sanity checks: after optimisation, f_M1 ≤ f_M2 and f_M1 ≤ f_M3 ≤ f_M4
should hold up to optimisation noise.  Violations are themselves informative
(they trigger re-optimisation with more restarts / different seeds).

These tests only run a small optimisation budget — they're meant as smoke
checks, not the full Phase-1/2 results.
"""
import pytest

from src.core.loss import evaluate
from src.monitors import M1Monitor, M2Monitor, M3Monitor, M4Monitor
from src.optim.gradient import train_one


@pytest.mark.slow
@pytest.mark.parametrize("n,k", [(11, 2), (11, 3)])
def test_nesting_smoke(n: int, k: int) -> None:
    """Train all four classes briefly; check the nesting ordering holds."""
    losses = {}
    for name, cls in [("M1", M1Monitor), ("M2", M2Monitor), ("M3", M3Monitor), ("M4", M4Monitor)]:
        m = cls(k=k, seed=42)
        res = train_one(m, n, lr=1e-2, steps=1500)
        losses[name] = res.final_loss

    # Nesting (with tolerance for optimisation noise).
    tol = 2e-3
    assert losses["M1"] <= losses["M2"] + tol, f"f_M1 > f_M2: {losses}"
    assert losses["M1"] <= losses["M3"] + tol, f"f_M1 > f_M3: {losses}"
    assert losses["M3"] <= losses["M4"] + tol, f"f_M3 > f_M4: {losses}"
