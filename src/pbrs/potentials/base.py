from __future__ import annotations

import abc
from typing import Protocol


class Potential(abc.ABC):
    """Abstract base class for potential functions Φ(s)."""

    @abc.abstractmethod
    def __call__(self, s: int) -> float:  # noqa: D401, E501
        """Return potential value Φ(s) for *state* ``s``."""

    # ------------------------------------------------------------------
    # Convenience: vectorised evaluation
    # ------------------------------------------------------------------
    def values(self, states: list[int]) -> list[float]:
        return [self(s) for s in states]

    # Optional: normalisation helper
    def normalise(self, *, target_min: float = 0.0, target_max: float = 1.0):
        raise NotImplementedError 