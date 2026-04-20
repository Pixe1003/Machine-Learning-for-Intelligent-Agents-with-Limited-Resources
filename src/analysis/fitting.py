"""
Bonus objective: fit candidate analytic forms to empirical f(n, k).

Candidate forms:
    C · k^{-α}
    C · exp(-β k)
    C · (n / k)^{-γ}

Selected by R² and AIC/BIC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from scipy.optimize import curve_fit


def _power_k(k, C, alpha):
    return C * k ** (-alpha)


def _exp_k(k, C, beta):
    return C * np.exp(-beta * k)


def _power_nk(nk, C, gamma):
    return C * nk ** (-gamma)


@dataclass
class FitResult:
    name: str
    params: tuple
    r2: float
    aic: float
    bic: float

    def __str__(self) -> str:
        return f"{self.name}: params={self.params}, R²={self.r2:.4f}, AIC={self.aic:.2f}, BIC={self.bic:.2f}"


def _fit(name: str, func: Callable, x: np.ndarray, y: np.ndarray, p0=None) -> FitResult:
    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
    y_hat = func(x, *popt)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    n = len(x)
    p = len(popt)
    aic = n * np.log(max(ss_res / n, 1e-12)) + 2 * p
    bic = n * np.log(max(ss_res / n, 1e-12)) + p * np.log(n)
    return FitResult(name=name, params=tuple(popt), r2=r2, aic=float(aic), bic=float(bic))


def fit_all(data: Dict[tuple, float]) -> List[FitResult]:
    """
    data : {(n, k): f_value} from an empirical sweep.
    Returns fits for k^{-α}, exp(-β k), (n/k)^{-γ}.
    """
    pts = [(n, k, f) for (n, k), f in data.items() if f > 0]
    n_arr = np.array([p[0] for p in pts], dtype=float)
    k_arr = np.array([p[1] for p in pts], dtype=float)
    f_arr = np.array([p[2] for p in pts], dtype=float)

    results: list[FitResult] = []
    try:
        results.append(_fit("C * k^-alpha", _power_k, k_arr, f_arr, p0=(0.5, 1.0)))
    except Exception as e:
        results.append(FitResult("C * k^-alpha", (), float("nan"), float("inf"), float("inf")))
    try:
        results.append(_fit("C * exp(-beta k)", _exp_k, k_arr, f_arr, p0=(0.5, 0.5)))
    except Exception as e:
        results.append(FitResult("C * exp(-beta k)", (), float("nan"), float("inf"), float("inf")))
    try:
        results.append(_fit("C * (n/k)^-gamma", _power_nk, n_arr / k_arr, f_arr, p0=(0.5, 1.0)))
    except Exception as e:
        results.append(FitResult("C * (n/k)^-gamma", (), float("nan"), float("inf"), float("inf")))

    return results
