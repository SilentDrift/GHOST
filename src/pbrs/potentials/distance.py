from __future__ import annotations

from typing import Dict

import numpy as np

from ..envs.graph_env import GraphMDP
from .base import Potential

__all__ = ["DistancePotential"]


class DistancePotential(Potential):
    """Potential Φ(s) = -d(s, goal) where *d* is the shortest-path length.

    If a state is unreachable from any goal, its distance is ∞ and the
    potential is set to a large negative value (``-max_dist``).
    """

    def __init__(self, mdp: GraphMDP) -> None:
        self._dist: Dict[int, float] = mdp.shortest_distances()
        # Determine finite distances to goals
        finite_dists = [d for d in self._dist.values() if np.isfinite(d)]
        if not finite_dists:
            raise ValueError("No states can reach any goal – unreachable graph?")
        self._max_dist = max(finite_dists)
        # Explicitly ensure potential at terminal nodes is zero
        self._dist.update({n: 0.0 for n in mdp.fail_nodes})

    def __call__(self, s: int) -> float:
        d = self._dist.get(s, np.inf)
        if np.isinf(d):
            return 0.0  # unreachable or fail nodes
        return -float(d) 