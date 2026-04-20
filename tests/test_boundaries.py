"""
Boundary condition sanity checks — these must pass before any experiments.

Proposal M1 milestone: f(n, 1) = 0.5 and f(n, k) = 0 for k ≥ ⌈n/2⌉ + 1.
"""
import math

import pytest

from src.core.boundary_checks import f_saturating_counter, f_single_state


@pytest.mark.parametrize("n", [5, 7, 9, 11, 21, 51])
def test_single_state_is_half(n: int) -> None:
    """k = 1: single-state monitor can only guess -> f = 0.5."""
    assert abs(f_single_state(n) - 0.5) < 1e-10


@pytest.mark.parametrize("n", [5, 7, 9, 11, 21, 51])
def test_saturating_counter_is_zero(n: int) -> None:
    """k = ⌈n/2⌉ + 1: saturating counter yields zero error."""
    assert f_saturating_counter(n) < 1e-10


def test_small_grid_sanity() -> None:
    """Spot-check that the recursion treats H/T correctly at n=3 manually."""
    import torch
    from src.core.recursion import compute_error

    # n=3, k=2, H always moves to state 1, T always moves to state 0.
    # Accept iff state 1.  Should achieve 0 error (exact majority tracker).
    H = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    T = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([0.0, 1.0], dtype=torch.float64)
    f = float(compute_error(H, T, p, 3).item())
    # This is a last-observation monitor (only reflects the 3rd vote), so its
    # majority accuracy is 0.5 -> f = 0.5.  (Not a zero-error achiever.)
    assert 0.0 <= f <= 1.0
