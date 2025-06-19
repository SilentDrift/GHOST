from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp


class GraphMDP:
    """Minimal deterministic graph MDP backed by a NetworkX directed graph.

    Nodes are expected to be *hashable* (int is recommended for performance).

    A transition corresponds to traversing an outgoing edge. Reward is sparse
    by default: +1 on reaching a *goal* node, −1 on reaching a *fail* node,
    and 0 elsewhere. When ``sparse=False`` users can override reward logic or
    attach a shaping wrapper.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        goal_nodes: Sequence[int],
        fail_nodes: Sequence[int],
        *,
        gamma: float = 0.99,
        sparse: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.G: nx.DiGraph = graph.copy()  # defensive copy to avoid side-effects
        self.goal_nodes: List[int] = list(goal_nodes)
        self.fail_nodes: List[int] = list(fail_nodes)
        self.gamma: float = gamma
        self.sparse: bool = sparse
        self.rng: random.Random = rng or random.Random()

        # Internal state
        self._state: Optional[int] = None
        self._done: bool = False

        # Cache frequently used sets for speed.
        self._goal_set = set(self.goal_nodes)
        self._fail_set = set(self.fail_nodes)
        self._terminal_set = self._goal_set | self._fail_set
        self._non_terminal_nodes = [n for n in self.G.nodes if n not in self._terminal_set]

        if not self._non_terminal_nodes:
            raise ValueError("Graph does not contain any non-terminal nodes!")

    # ---------------------------------------------------------------------
    # MDP API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, start: Optional[int] = None) -> int:
        """Reset environment and return initial state.

        Args:
            seed: optional seed to re-seed the underlying RNG for this *call*.
            start: explicit start node. Must be a non-terminal node if given.
        """
        if seed is not None:
            self.rng.seed(seed)
        if start is not None:
            if start in self._terminal_set:
                raise ValueError("start state cannot be terminal (goal/fail)")
            if start not in self.G:
                raise KeyError("start node not found in graph")
            self._state = start
        else:
            self._state = self.rng.choice(self._non_terminal_nodes)
        self._done = False
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Apply *action* and return (next_state, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step().")
        if self._done:
            raise RuntimeError("Episode has terminated. Call reset().")

        # Validate action
        if not self.G.has_edge(self._state, action):
            raise ValueError(f"Invalid action {action} from state {self._state}.")

        next_state = action  # deterministic transition
        self._state = next_state
        self._done = next_state in self._terminal_set

        # ------------------------------------------------------------------
        # Reward logic
        # ------------------------------------------------------------------
        reward: float = 0.0
        if next_state in self._goal_set:
            reward = 1.0  # success
        elif next_state in self._fail_set:
            reward = -1.0  # failure
        elif not self.sparse:
            # Dense reward can be customised externally (e.g., shaping wrapper).
            reward = 0.0
        # If sparse=True and not terminal we already default to 0.

        info: Dict[str, Any] = {}
        return next_state, reward, self._done, info

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def shortest_distances(self) -> Dict[int, float]:
        """Return *shortest path length* to the closest goal node for *all* nodes.

        Uses a multi-source BFS. Nodes unreachable from any goal receive ∞.
        """
        # Reverse graph to search from goals outward.
        rev = self.G.reverse(copy=False)
        dist = {n: np.inf for n in rev.nodes()}
        frontier = list(self.goal_nodes)
        for g in frontier:
            dist[g] = 0.0
        while frontier:
            current = frontier.pop(0)
            for pred in rev.neighbors(current):
                if dist[pred] == np.inf:
                    dist[pred] = dist[current] + 1
                    frontier.append(pred)
        return dist

    def laplacian_matrix(self) -> sp.csr_matrix:
        """Return the (out-degree) Laplacian matrix L = D − A as CSR sparse."""
        # For directed graphs we follow the *out-degree* convention.
        n = self.G.number_of_nodes()
        A = nx.to_scipy_sparse_array(self.G, format="csr", dtype=np.float64)
        degree_out = np.asarray(A.sum(axis=1)).ravel()
        D = sp.diags(degree_out, format="csr")
        return D - A

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.G.number_of_nodes()

    @property
    def n_actions(self) -> int:
        """Upper-bound on number of actions (uses node id as action)."""
        return self.n_states

    @property
    def state(self) -> int:
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset().")
        return self._state 