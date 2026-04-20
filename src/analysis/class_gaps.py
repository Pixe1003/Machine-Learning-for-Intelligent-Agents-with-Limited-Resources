"""
Class-gap diagnostics:
    Δ^acc(n,k)     = f_M2 - f_M1
    Δ^count(n,k)   = f_M3 - f_M1
    Δ^minimal(n,k) = f_M4 - f_M3

Used to evaluate hypotheses H1 (counter-equivalence) and H5 (soft acceptance).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassGaps:
    n: int
    k: int
    f_M1: float
    f_M2: float
    f_M3: float
    f_M4: float

    @property
    def delta_acc(self) -> float:
        return self.f_M2 - self.f_M1

    @property
    def delta_count(self) -> float:
        return self.f_M3 - self.f_M1

    @property
    def delta_minimal(self) -> float:
        return self.f_M4 - self.f_M3

    @property
    def delta_count_relative(self) -> float:
        return self.delta_count / max(self.f_M1, 1e-12)

    def nesting_violations(self) -> list[str]:
        """Return any violations of f_M1 ≤ f_M2 and f_M1 ≤ f_M3 ≤ f_M4."""
        v: list[str] = []
        if self.f_M1 > self.f_M2 + 1e-8:
            v.append(f"f_M1={self.f_M1:.6f} > f_M2={self.f_M2:.6f}")
        if self.f_M1 > self.f_M3 + 1e-8:
            v.append(f"f_M1={self.f_M1:.6f} > f_M3={self.f_M3:.6f}")
        if self.f_M3 > self.f_M4 + 1e-8:
            v.append(f"f_M3={self.f_M3:.6f} > f_M4={self.f_M4:.6f}")
        return v

    def summary(self) -> str:
        return (
            f"(n={self.n}, k={self.k})  "
            f"f_M1={self.f_M1:.5f}  f_M2={self.f_M2:.5f}  "
            f"f_M3={self.f_M3:.5f}  f_M4={self.f_M4:.5f}  |  "
            f"Δ^acc={self.delta_acc:+.5f}  Δ^count={self.delta_count:+.5f}  "
            f"Δ^count/f_M1={self.delta_count_relative:.2%}"
        )
