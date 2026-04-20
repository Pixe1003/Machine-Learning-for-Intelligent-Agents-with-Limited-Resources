"""Abstract Monitor base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn

from src.core.recursion import DEFAULT_DTYPE


class MonitorBase(nn.Module, ABC):
    """
    All learned monitors inherit from this class.

    Subclasses must implement:
        - transitions(n) -> (H, T)       row-stochastic, (k,k) or (n,k,k)
        - acceptance()   -> (k,)         in [0, 1]

    They may override:
        - initial() -> (k,) in Δ^k       (default: δ_0)
    """

    def __init__(self, k: int, dtype=DEFAULT_DTYPE, device=None):
        super().__init__()
        self.k = k
        self._dtype_ = dtype
        self._device_ = device

    @abstractmethod
    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        """Return (H, T) row-stochastic transition tensors."""

    @abstractmethod
    def acceptance(self) -> Tensor:
        """Return per-state acceptance probability in [0, 1]."""

    def initial(self) -> Tensor:
        pi0 = torch.zeros(self.k, dtype=self._dtype_, device=self._device_)
        pi0[0] = 1.0
        return pi0

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return f"k={self.k}, n_params={self.n_parameters()}"
