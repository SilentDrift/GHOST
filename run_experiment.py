import json
import sys

sys.path.append('src')

import pbrs.envs as penv
import pbrs.potentials as pp
import pbrs.shaping as ps
import pbrs.agents as pa
import pbrs.training as pt


def main():
    mdp, _ = penv.er_graph(n=20, p=0.2, seed=0)
    env = penv.GraphEnv(mdp)
    pot = pp.HarmonicPotential(mdp)
    env = ps.RewardShaper(env, pot, gamma=0.99)
    agent = pa.TabularQAgent(env.observation_space.n, env.action_space.n)
    trainer = pt.Trainer(env, agent, train_episodes=200, log_every=0, seed=0)
    summary = trainer.run()
    print(json.dumps(summary))


if __name__ == "__main__":
    main() 