from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Potential

__all__ = ["ComboPotential"]


class ComboPotential(Potential):
    """Linear combination Φ = Σ_i w_i Φ_i."""

    def __init__(self, potentials: Sequence[Potential], weights: Sequence[float] | None = None):
        if not potentials:
            raise ValueError("Need at least one sub-potential")
        if weights is None:
            weights = [1.0] * len(potentials)
        if len(potentials) != len(weights):
            raise ValueError("Number of weights must match potentials")
        self._potentials = list(potentials)
        self._weights = np.array(weights, dtype=float)

    def __call__(self, s: int) -> float:
        return float(sum(w * p(s) for w, p in zip(self._weights, self._potentials))) 