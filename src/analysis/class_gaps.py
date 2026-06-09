"""Class-gap diagnostics under the presentation hierarchy.

Presentation names:
    M1       = unconstrained transitions with soft acceptance
    M1-hard  = unconstrained transitions with hard report-time acceptance
    M2       = state-dependent stochastic counting monitor
    M3       = minimal counting monitor

The clear nesting story is M1, M2, M3.  M1-hard is tracked separately because
hard acceptance is weaker than M1 but not part of the M1/M2/M3 counting
hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassGaps:
    """Class-gap diagnostics under the supervisor's renamed hierarchy.

    Native eval modes (what each class is selected by during training):
        f_M1, f_M2, f_M3 : soft (continuous sigmoid acceptance)
        f_M1_hard        : hard (binary thresholded acceptance)

    The ``*_alt`` fields hold the *cross-eval* values: each monitor evaluated
    under the opposite acceptance mode.  When supplied, this lets us compute
    ``delta_*_soft`` and ``delta_*_hard`` using a CONSISTENT eval mode across
    classes, avoiding the cross-criterion artefact that can make
    ``delta_hard`` (legacy mixed eval) appear negative.

    Mapping:
        f_M1_alt        = soft-trained M1 monitor, evaluated under hard
        f_M1_hard_alt   = hard-trained M1-hard monitor, evaluated under soft
        f_M2_alt        = soft-trained M2 monitor, evaluated under hard
        f_M3_alt        = soft-trained M3 monitor, evaluated under hard
    """

    n: int
    k: int
    f_M1: float
    f_M1_hard: float
    f_M2: float
    f_M3: float
    # Optional dual-eval cross-values. Default None preserves backward-compat.
    f_M1_alt: float | None = None
    f_M1_hard_alt: float | None = None
    f_M2_alt: float | None = None
    f_M3_alt: float | None = None

    # ---------- legacy / mixed-eval gaps (kept for backward compat) ----------
    @property
    def delta_hard(self) -> float:
        """f_M1-hard (native hard) - f_M1 (native soft).

        Mixed-eval; can be negative for purely numerical reasons. Use
        delta_acc_soft / delta_acc_hard for the methodologically clean
        versions.
        """
        return self.f_M1_hard - self.f_M1

    @property
    def delta_count(self) -> float:
        """Δ^count under soft eval (the canonical H1 quantity)."""
        return self.f_M2 - self.f_M1

    @property
    def delta_minimal(self) -> float:
        """Δ^minimal under soft eval."""
        return self.f_M3 - self.f_M2

    @property
    def delta_count_relative(self) -> float:
        return self.delta_count / max(self.f_M1, 1e-12)

    # ---------- consistent-eval (dual-eval) gaps ----------------------------
    @property
    def delta_acc_soft(self) -> float | None:
        """Δ^acc under consistent SOFT eval: M1-hard.soft - M1.soft.

        Should be ≥ 0 by nesting (M1 ⊇ M1-hard in the soft-eval landscape).
        """
        if self.f_M1_hard_alt is None:
            return None
        return self.f_M1_hard_alt - self.f_M1

    @property
    def delta_acc_hard(self) -> float | None:
        """Δ^acc under consistent HARD eval: M1-hard.hard - M1.hard.

        Should be ≤ 0 (M1-hard's best hard monitor is at most as good as
        re-evaluating M1's soft-selected monitor under hard); a positive
        value flags that M1's optimisation didn't reach a binary configuration
        as good as M1-hard's.
        """
        if self.f_M1_alt is None:
            return None
        return self.f_M1_hard - self.f_M1_alt

    @property
    def delta_count_hard(self) -> float | None:
        """Δ^count under consistent HARD eval: M2.hard - M1.hard."""
        if self.f_M2_alt is None or self.f_M1_alt is None:
            return None
        return self.f_M2_alt - self.f_M1_alt

    @property
    def delta_minimal_hard(self) -> float | None:
        """Δ^minimal under consistent HARD eval: M3.hard - M2.hard."""
        if self.f_M3_alt is None or self.f_M2_alt is None:
            return None
        return self.f_M3_alt - self.f_M2_alt

    # ---------- diagnostics --------------------------------------------------
    def nesting_violations(self) -> list[str]:
        """Return any violations of f_M1 <= f_M2 <= f_M3 (all under soft eval)."""
        v: list[str] = []
        if self.f_M1 > self.f_M2 + 1e-8:
            v.append(f"f_M1={self.f_M1:.6f} > f_M2={self.f_M2:.6f}")
        if self.f_M2 > self.f_M3 + 1e-8:
            v.append(f"f_M2={self.f_M2:.6f} > f_M3={self.f_M3:.6f}")
        return v

    def summary(self) -> str:
        base = (
            f"(n={self.n}, k={self.k})  "
            f"f_M1={self.f_M1:.5f}  f_M1-hard={self.f_M1_hard:.5f}  "
            f"f_M2={self.f_M2:.5f}  f_M3={self.f_M3:.5f}  |  "
            f"delta_hard={self.delta_hard:+.5f}  delta_count={self.delta_count:+.5f}  "
            f"delta_count/f_M1={self.delta_count_relative:.2%}"
        )
        if self.delta_acc_soft is not None:
            base += (
                f"  |  delta_acc_soft={self.delta_acc_soft:+.5f}  "
                f"delta_acc_hard={self.delta_acc_hard:+.5f}"
            )
        return base
