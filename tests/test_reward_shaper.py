import math
import networkx as nx
from pbrs.envs.graph_env import GraphMDP
from pbrs.envs.wrappers import GraphEnv
from pbrs.shaping import RewardShaper
from pbrs.potentials.base import Potential

class SimplePotential(Potential):
    def __call__(self, s: int) -> float:
        # Φ(1)=0 for goal, Φ(0)=1 for start
        return {0: 1.0, 1: 0.0}.get(s, 0.0)

def test_shaping_single_step():
    g = nx.DiGraph()
    g.add_edge(0, 1)
    mdp = GraphMDP(g, goal_nodes=[1], fail_nodes=[], gamma=0.9)
    env = GraphEnv(mdp)
    potential = SimplePotential()
    shaped = RewardShaper(env, potential, gamma=mdp.gamma)

    obs, _ = shaped.reset(seed=123, options={"start": 0})
    assert obs == 0
    next_obs, reward, terminated, truncated, info = shaped.step(1)

    assert terminated and not truncated
    expected_F = mdp.gamma * potential(next_obs) - potential(obs)
    assert math.isclose(info["shaping"], expected_F)
