from __future__ import annotations

from typing import Dict

import numpy as np

from ..envs.graph_env import GraphMDP
from .base import Potential

__all__ = ["DistancePotential"]


class DistancePotential(Potential):
    """Potential Φ(s) = -d(s, goal) where *d* is the shortest-path length.

    States that cannot reach any goal receive a potential of ``-max_dist``
    (the worst finite distance observed). Terminal states remain at zero.
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
            # Unreachable states are assigned the worst potential
            return -float(self._max_dist)
        return -float(d)
