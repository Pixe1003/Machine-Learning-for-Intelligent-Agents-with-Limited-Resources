"""
Internal consistency checks for the forward recursion.
"""
import math

import pytest
import torch

from src.core.recursion import _log_binomial_coeffs, compute_error, state_distribution


@pytest.mark.parametrize("n,k", [(5, 2), (9, 3), (11, 4)])
def test_state_distribution_sums_to_one(n: int, k: int) -> None:
    """For every h, π^(h)_n should be a probability distribution over states."""
    torch.manual_seed(0)
    H = torch.softmax(torch.randn(k, k, dtype=torch.float64), dim=-1)
    T = torch.softmax(torch.randn(k, k, dtype=torch.float64), dim=-1)
    pi = state_distribution(H, T, n)  # (n+1, k)
    row_sums = pi.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-9)


@pytest.mark.parametrize("n", [5, 7, 11, 21])
def test_log_binomial_coeffs_matches_scipy(n: int) -> None:
    from scipy.special import gammaln

    log_coeffs = _log_binomial_coeffs(n)
    expected = [gammaln(n + 1) - gammaln(h + 1) - gammaln(n - h + 1) for h in range(n + 1)]
    for a, b in zip(log_coeffs.tolist(), expected):
        assert abs(a - b) < 1e-10


def test_error_is_differentiable() -> None:
    """Gradient should flow through H, T, and p."""
    n, k = 11, 3
    W_H = torch.randn(k, k, dtype=torch.float64, requires_grad=True)
    W_T = torch.randn(k, k, dtype=torch.float64, requires_grad=True)
    omega = torch.randn(k, dtype=torch.float64, requires_grad=True)

    H = torch.softmax(W_H, dim=-1)
    T = torch.softmax(W_T, dim=-1)
    p = torch.sigmoid(omega)

    f = compute_error(H, T, p, n)
    f.backward()
    assert W_H.grad is not None and torch.any(W_H.grad != 0)
    assert W_T.grad is not None and torch.any(W_T.grad != 0)
    assert omega.grad is not None and torch.any(omega.grad != 0)


def test_error_bounds() -> None:
    """f(n, k) ∈ [0, 1] for any valid (H, T, p)."""
    n, k = 9, 3
    H = torch.softmax(torch.randn(k, k, dtype=torch.float64), dim=-1)
    T = torch.softmax(torch.randn(k, k, dtype=torch.float64), dim=-1)
    p = torch.sigmoid(torch.randn(k, dtype=torch.float64))
    f = float(compute_error(H, T, p, n).item())
    assert 0.0 <= f <= 1.0


def test_n_must_be_odd() -> None:
    H = torch.eye(2, dtype=torch.float64)
    T = torch.eye(2, dtype=torch.float64)
    p = torch.tensor([0.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        compute_error(H, T, p, 4)
