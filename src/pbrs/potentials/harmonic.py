from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..envs.graph_env import GraphMDP
from .base import Potential

__all__ = ["HarmonicPotential"]


class HarmonicPotential(Potential):
    """Harmonic potential obtained by solving the discrete Laplace equation.

    By default we set Φ(goal)=0 and Φ(fail)=1 (Dirichlet boundary values). The
    solution inside the domain satisfies ΔΦ = 0. The resulting values are
    normalised in [0, 1]. If *scale* is provided, we linearly map to that
    interval instead (e.g. scale=(-1, 1)).
    """

    def __init__(
        self,
        mdp: GraphMDP,
        *,
        solver: str = "cg",
        fail_value: float = 1.0,
        goal_value: float = 0.0,
        scale: tuple[float, float] | None = None,
        tol: float = 1e-8,
        maxiter: int = 10_000,
    ) -> None:
        if solver not in {"cg", "spsolve"}:
            raise ValueError("solver must be 'cg' or 'spsolve'")
        self._solver = solver
        self._mdp = mdp

        # Build Laplacian
        L = mdp.laplacian_matrix().tocsr()
        n = mdp.n_states

        # Boundary conditions
        goal = list(mdp.goal_nodes)
        fail = list(mdp.fail_nodes)
        fixed_nodes: list[int] = goal + fail
        free_nodes: list[int] = [i for i in range(n) if i not in fixed_nodes]

        if not free_nodes:
            raise ValueError("All nodes are terminal – nothing to solve")

        # Build RHS
        phi_fixed = np.zeros(len(fixed_nodes))
        # goal nodes 0, fail nodes fail_value
        for i, node in enumerate(fixed_nodes):
            if node in fail:
                phi_fixed[i] = fail_value
            else:
                phi_fixed[i] = goal_value

        # Extract submatrices
        L_ff = L[free_nodes][:, free_nodes]
        L_fi = L[free_nodes][:, fixed_nodes]
        b = -L_fi.dot(phi_fixed)

        # Solve
        if self._solver == "cg":
            try:
                # SciPy <1.15 uses `tol`, newer uses `rtol`.
                x, info = spla.cg(L_ff, b, tol=tol, maxiter=maxiter)
            except TypeError:
                x, info = spla.cg(L_ff, b, rtol=tol, maxiter=maxiter)
            if info != 0:
                # Fall back to direct solver
                x = spla.spsolve(L_ff, b)
        else:
            x = spla.spsolve(L_ff, b)

        phi = np.empty(n, dtype=float)
        for idx, node in enumerate(free_nodes):
            phi[node] = x[idx]
        for idx, node in enumerate(fixed_nodes):
            phi[node] = phi_fixed[idx]

        # Normalise to [0,1]
        min_phi, max_phi = phi.min(), phi.max()
        if max_phi - min_phi > 0:
            phi = (phi - min_phi) / (max_phi - min_phi)
        else:
            # degenerate but okay
            phi[:] = 0.0

        if scale is not None:
            a, b = scale
            phi = a + (b - a) * phi  # linear rescale

        self._phi = phi

    # ------------------------------------------------------------------
    # Potential interface
    # ------------------------------------------------------------------
    def __call__(self, s: int) -> float:
        return float(self._phi[s])

    # Convenience property to expose full vector
    @property
    def values_vector(self) -> np.ndarray:
        return self._phi.copy() 