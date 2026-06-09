"""
Nesting sanity checks under the presentation hierarchy:

    M1 <= M2 <= M3

M1-hard is tracked separately because it is a hard-acceptance variant of the
unconstrained model, not part of the counting hierarchy.
"""
import pytest

from src.monitors import M1HardMonitor, M1Monitor, M2Monitor, M3Monitor
from src.optim.gradient import train_one


@pytest.mark.slow
@pytest.mark.parametrize("n,k", [(11, 2), (11, 3)])
def test_nesting_smoke(n: int, k: int) -> None:
    losses = {}
    for name, cls in [
        ("M1", M1Monitor),
        ("M1-hard", M1HardMonitor),
        ("M2", M2Monitor),
        ("M3", M3Monitor),
    ]:
        m = cls(k=k, seed=42)
        res = train_one(m, n, lr=1e-2, steps=1500)
        losses[name] = res.final_loss

    tol = 2e-3
    assert losses["M1"] <= losses["M2"] + tol, f"f_M1 > f_M2: {losses}"
    assert losses["M2"] <= losses["M3"] + tol, f"f_M2 > f_M3: {losses}"
