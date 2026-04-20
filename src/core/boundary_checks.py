"""
Boundary-condition sanity checks for f(n, k).

From the proposal:
    f(n, 1) = 0.5                    (single state: random guess)
    f(n, k) = 0  for k ≥ ⌈n/2⌉ + 1   (saturating counter is exact)

These are *independent* of any training — a correctly implemented recursion
must produce them for any valid parameterisation of the saturating-counter
monitor.  Passing both constitutes the M1 milestone described in the proposal.
"""
from __future__ import annotations

import math

import torch

from src.core.recursion import compute_error


def f_single_state(n: int, accept_prob: float = 0.5) -> float:
    """
    k=1 monitor: p(s_0) is the only free parameter.
    By symmetry the minimiser is p=0.5, yielding f = 0.5.
    """
    H = torch.eye(1, dtype=torch.float64)
    T = torch.eye(1, dtype=torch.float64)
    p = torch.tensor([accept_prob], dtype=torch.float64)
    return float(compute_error(H, T, p, n).item())


def f_saturating_counter(n: int) -> float:
    """
    Exact majority monitor for k = ⌈n/2⌉ + 1 states.

    States encode the count of Heads so far, capped at k-1.  Accept iff
    final count exceeds n/2 (equivalently, state ≥ ⌈n/2⌉).
    """
    k = math.ceil(n / 2) + 1

    # H transitions: state s -> min(s+1, k-1) deterministically.
    H = torch.zeros(k, k, dtype=torch.float64)
    for s in range(k):
        H[s, min(s + 1, k - 1)] = 1.0

    # T transitions: stay.
    T = torch.eye(k, dtype=torch.float64)

    # Accept iff h > n/2 (n is odd so threshold = ⌈n/2⌉).
    threshold = math.ceil(n / 2)
    p = torch.zeros(k, dtype=torch.float64)
    p[threshold:] = 1.0

    return float(compute_error(H, T, p, n).item())


def run_all_boundary_checks(ns=(5, 7, 9, 11, 21), atol: float = 1e-10) -> dict:
    """Return a dict of boundary-condition results."""
    results: dict[str, dict] = {"f_n_1": {}, "f_saturating": {}}
    for n in ns:
        v1 = f_single_state(n)
        v2 = f_saturating_counter(n)
        results["f_n_1"][n] = v1
        results["f_saturating"][n] = v2
        assert abs(v1 - 0.5) < atol, f"f({n},1)={v1} should be 0.5"
        assert v2 < atol, f"f_saturating({n})={v2} should be 0"
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run_all_boundary_checks(), indent=2))
