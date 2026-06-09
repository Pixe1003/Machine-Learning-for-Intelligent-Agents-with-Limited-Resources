"""Transition-matrix heat-maps (H*, T*) for the structural analysis section.

Two entry points:
    plot_heatmap                -> single-class H/T pair (legacy, still useful)
    plot_class_comparison_heatmap -> one figure covering all of M1-M4 (+ dynamic)
                                     which is the preferred per-(n, k) artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_2d(mat: Tensor) -> np.ndarray:
    """For static tensors return (k, k); for dynamic (n, k, k) return the
    middle-step slice, which is the most informative single snapshot."""
    arr = mat.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    return arr


def _annotate(ax, mat: np.ndarray, fontsize: int = 7) -> None:
    k = mat.shape[0]
    for s in range(k):
        for sp in range(k):
            v = mat[s, sp]
            if v > 0.01:
                ax.text(
                    sp, s, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=fontsize,
                )


# ---------------------------------------------------------------------------
# single-class view (kept for backwards compatibility / debugging)
# ---------------------------------------------------------------------------

def plot_heatmap(H: Tensor, T: Tensor, title: str, out_path: str | Path) -> None:
    """Side-by-side H / T heat-maps with cell values annotated."""
    H_np = _to_2d(H)
    T_np = _to_2d(T)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, mat, name in zip(axes, (H_np, T_np), ("H (Head)", "T (Tail)")):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("next state s'")
        ax.set_ylabel("state s")
        _annotate(ax, mat)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_heatmap(H: Tensor, T: Tensor, title: str, out_path: str | Path,
                         step_indices=None) -> None:
    """Film-strip view for dynamic monitors: H_i, T_i at a few representative i."""
    n = H.shape[0]
    if step_indices is None:
        step_indices = np.linspace(0, n - 1, 4, dtype=int)
    fig, axes = plt.subplots(2, len(step_indices), figsize=(3 * len(step_indices), 6))
    for col, i in enumerate(step_indices):
        for row, (mat, name) in enumerate([(H[i], "H"), (T[i], "T")]):
            ax = axes[row, col]
            ax.imshow(mat.detach().cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"{name}  i={int(i)}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# consolidated class-comparison view (this is the figure we actually use)
# ---------------------------------------------------------------------------

def plot_class_comparison_heatmap(
    monitors: Mapping[str, object],
    n: int,
    k: int,
    out_path: str | Path,
    *,
    title: str | None = None,
    annotate: bool | None = None,
    diagnostics: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """
    Single figure showing H and T for every class at fixed (n, k).

    Layout: len(monitors) rows x 2 cols.  One shared colorbar.
    If `diagnostics` is provided (class_name -> {tridiagonality, transition_entropy_mean}),
    those numbers are printed in each row's y-label for quick reading.
    """
    classes = list(monitors.keys())
    nrows = len(classes)
    if annotate is None:
        annotate = k <= 6  # noisy for large k

    fig, axes = plt.subplots(nrows, 2, figsize=(6.5, 2.4 * nrows + 0.3), squeeze=False)
    im = None
    for r, name in enumerate(classes):
        H, T = monitors[name].transitions(n)
        H_np, T_np = _to_2d(H), _to_2d(T)
        for c, (mat, label) in enumerate([(H_np, "H (Head)"), (T_np, "T (Tail)")]):
            ax = axes[r, c]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="equal")
            if r == 0:
                ax.set_title(label)
            if c == 0:
                tag = name
                if diagnostics and name in diagnostics:
                    d = diagnostics[name]
                    tri = d.get("tridiagonality")
                    ent = d.get("transition_entropy_mean")
                    bits = [name]
                    if tri is not None:
                        bits.append(rf"$\tau={tri:.3f}$")
                    if ent is not None:
                        bits.append(rf"$\bar H={ent:.2f}$")
                    tag = "\n".join(bits)
                ax.set_ylabel(tag, fontsize=9)
            ax.set_xticks(range(k))
            ax.set_yticks(range(k))
            ax.tick_params(labelsize=7)
            if annotate:
                _annotate(ax, mat, fontsize=6)

    if im is not None:
        fig.subplots_adjust(right=0.87)
        cbar_ax = fig.add_axes([0.89, 0.12, 0.02, 0.76])
        fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(title or f"Class comparison (n={n}, k={k})", fontsize=11)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
