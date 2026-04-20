"""
Exhaustive grid search for small (n, k) to provide a certified lower bound.

For n ∈ {5, 7, 9, 11}, k ∈ {2, 3}:
  - Each row of H, T lies on the (k-1)-simplex; we grid-search at resolution
    res (default 0.05) then enumerate all 2^k binary acceptance sets F.

Used to validate gradient-descent and GA solutions in Phase 1.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch

from src.core.recursion import compute_error


@dataclass
class ExhaustiveResult:
    n: int
    k: int
    f_star: float
    H: torch.Tensor
    T: torch.Tensor
    p: torch.Tensor


def _simplex_grid(k: int, res: float) -> List[Tuple[float, ...]]:
    """Enumerate points on the (k-1)-simplex at resolution `res`."""
    steps = round(1.0 / res)
    pts: list[tuple[float, ...]] = []

    def _rec(remaining_steps: int, left: int, prefix: tuple[int, ...]) -> None:
        if left == 1:
            pts.append(prefix + (remaining_steps,))
            return
        for s in range(remaining_steps + 1):
            _rec(remaining_steps - s, left - 1, prefix + (s,))

    _rec(steps, k, tuple())
    return [tuple(v / steps for v in p) for p in pts]


def _rows(k: int, res: float) -> List[torch.Tensor]:
    return [torch.tensor(p, dtype=torch.float64) for p in _simplex_grid(k, res)]


def exhaustive_search(n: int, k: int, res: float = 0.1, verbose: bool = True) -> ExhaustiveResult:
    """
    Exhaustively enumerate (H, T) on a simplex grid and all binary F.

    WARNING: explodes as (k, 1/res) grow.  Feasible for k ≤ 3, res ≥ 0.05.
    """
    rows = _rows(k, res)
    if verbose:
        print(f"  (n={n}, k={k}, res={res}): {len(rows)} rows per matrix -> {len(rows)**(2*k)} combos")

    best_f = float("inf")
    best_H = best_T = best_p = None

    for H_rows in itertools.product(rows, repeat=k):
        H = torch.stack(H_rows)
        for T_rows in itertools.product(rows, repeat=k):
            T = torch.stack(T_rows)
            for F_mask in range(2 ** k):
                p = torch.tensor([(F_mask >> s) & 1 for s in range(k)], dtype=torch.float64)
                f = float(compute_error(H, T, p, n).item())
                if f < best_f:
                    best_f = f
                    best_H, best_T, best_p = H.clone(), T.clone(), p.clone()

    assert best_H is not None
    return ExhaustiveResult(n=n, k=k, f_star=best_f, H=best_H, T=best_T, p=best_p)
