import os
import sys
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pbrs.envs.graph_env import GraphMDP
from pbrs.potentials.distance import DistancePotential


def test_distance_potential_unreachable():
    g = nx.DiGraph()
    g.add_edge(0, 1)
    g.add_node(2)
    g.add_node(3)  # unreachable non-terminal
    mdp = GraphMDP(g, goal_nodes=[1], fail_nodes=[2], gamma=0.9)
    pot = DistancePotential(mdp)
    assert pot(1) == 0.0
    assert pot(2) == 0.0  # fail node
    assert pot(0) == -1.0  # distance 1 to goal
    assert pot(3) == -1.0  # unreachable uses -max_dist
