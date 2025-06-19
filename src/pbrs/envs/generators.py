from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from .graph_env import GraphMDP

__all__ = [
    "er_graph",
    "ba_graph",
    "grid_maze",
]


DEFAULT_GAMMA = 0.99


def _pick_terminals(
    G: nx.DiGraph, *, rng: np.random.Generator | None = None
) -> Tuple[List[int], List[int]]:
    """Sample *one* goal and *one* fail node uniformly at random.

    Ensures sampled nodes are distinct and have non-zero in/out degree to avoid
    degenerate graphs.
    """
    rng = rng or np.random.default_rng()
    candidates = [n for n in G.nodes if G.out_degree(n) + G.in_degree(n) > 0]
    if len(candidates) < 2:
        raise ValueError("Need at least two viable nodes to sample terminals.")
    goal = rng.choice(candidates)
    fail = goal
    while fail == goal:
        fail = rng.choice(candidates)
    return [int(goal)], [int(fail)]


def er_graph(
    n: int,
    p: float,
    *,
    directed: bool = True,
    gamma: float = DEFAULT_GAMMA,
    sparse: bool = True,
    seed: int | None = None,
    **kw: Any,
) -> Tuple[GraphMDP, Dict[str, Any]]:
    """Erdős–Rényi G(n, p) generator returning a ``GraphMDP``.

    If ``directed`` is True, the underlying undirected graph is first sampled
    then each edge is given a random direction.
    """
    rng = np.random.default_rng(seed)
    G_u = nx.gnp_random_graph(n, p, seed=seed)
    if directed:
        G_d = nx.DiGraph()
        for u, v in G_u.edges():
            if rng.random() < 0.5:
                G_d.add_edge(u, v)
            else:
                G_d.add_edge(v, u)
        G_tmp = G_d
    else:
        G_tmp = G_u.to_directed()

    # Relabel nodes to consecutive ints starting at 0 for table-based agents
    G = nx.convert_node_labels_to_integers(G_tmp, ordering="sorted")

    goal_nodes, fail_nodes = _pick_terminals(G, rng=rng)
    mdp = GraphMDP(G, goal_nodes, fail_nodes, gamma=gamma, sparse=sparse)

    meta = {
        "type": "ER",
        "n": n,
        "p": p,
        "directed": directed,
        "goal_nodes": goal_nodes,
        "fail_nodes": fail_nodes,
        "seed": seed,
    }
    return mdp, meta


def ba_graph(
    n: int,
    m: int,
    *,
    gamma: float = DEFAULT_GAMMA,
    sparse: bool = True,
    seed: int | None = None,
    **kw: Any,
) -> Tuple[GraphMDP, Dict[str, Any]]:
    """Barabási–Albert preferential attachment graph."""
    rng = np.random.default_rng(seed)
    G_u = nx.barabasi_albert_graph(n, m, seed=seed)
    G = G_u.to_directed()

    # Relabel nodes to consecutive ints starting at 0 for table-based agents
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    goal_nodes, fail_nodes = _pick_terminals(G, rng=rng)
    mdp = GraphMDP(G, goal_nodes, fail_nodes, gamma=gamma, sparse=sparse)

    meta = {
        "type": "BA",
        "n": n,
        "m": m,
        "goal_nodes": goal_nodes,
        "fail_nodes": fail_nodes,
        "seed": seed,
    }
    return mdp, meta


def grid_maze(
    height: int,
    width: int,
    wall_prob: float,
    *,
    four_neighbour: bool = True,
    gamma: float = DEFAULT_GAMMA,
    sparse: bool = True,
    seed: int | None = None,
    **kw: Any,
) -> Tuple[GraphMDP, Dict[str, Any]]:
    """2-D grid maze with random walls.

    Nodes are labelled by integer coordinates flattened via ``y * width + x``.
    An edge exists between adjacent cells unless either side is a wall.
    """
    rng = np.random.default_rng(seed)

    H, W = height, width
    # Helper to convert (y, x) -> id
    def nid(y: int, x: int) -> int:
        return y * W + x

    walls = rng.random((H, W)) < wall_prob
    G = nx.DiGraph()

    # Add nodes first
    for y in range(H):
        for x in range(W):
            idx = nid(y, x)
            G.add_node(idx, pos=(x, y), is_wall=bool(walls[y, x]))

    # Add edges
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] if four_neighbour else [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for y in range(H):
        for x in range(W):
            if walls[y, x]:
                continue
            for dy, dx in directions:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and not walls[ny, nx_]:
                    G.add_edge(nid(y, x), nid(ny, nx_))

    # Remove wall nodes (isolated) to avoid buggy transitions
    wall_nodes = [nid(y, x) for y in range(H) for x in range(W) if walls[y, x]]
    G.remove_nodes_from(wall_nodes)

    # Relabel nodes to consecutive ints starting at 0 for table-based agents
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    goal_nodes, fail_nodes = _pick_terminals(G, rng=rng)
    mdp = GraphMDP(G, goal_nodes, fail_nodes, gamma=gamma, sparse=sparse)

    meta: Dict[str, Any] = {
        "type": "GridMaze",
        "height": height,
        "width": width,
        "wall_prob": wall_prob,
        "goal_nodes": goal_nodes,
        "fail_nodes": fail_nodes,
        "seed": seed,
    }
    return mdp, meta 