from __future__ import annotations

import argparse
from typing import Any

import pbrs.envs as penv
from pbrs.potentials import DistancePotential, HarmonicPotential
from pbrs.shaping import RewardShaper
from pbrs.agents import TabularQAgent
from pbrs.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on graph MDP")
    sub = parser.add_subparsers(dest="graph", required=True)

    er = sub.add_parser("er")
    er.add_argument("--n", type=int, default=100, help="Number of nodes")
    er.add_argument("--p", type=float, default=0.05, help="Edge probability")

    ba = sub.add_parser("ba")
    ba.add_argument("--n", type=int, default=100)
    ba.add_argument("--m", type=int, default=2)

    grid = sub.add_parser("grid")
    grid.add_argument("--height", type=int, default=10)
    grid.add_argument("--width", type=int, default=10)
    grid.add_argument("--wall_prob", type=float, default=0.2)

    # shared options
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--potential", choices=["distance", "harmonic"], default="distance")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    if args.graph == "er":
        mdp, _ = penv.er_graph(n=args.n, p=args.p, seed=args.seed)
    elif args.graph == "ba":
        mdp, _ = penv.ba_graph(n=args.n, m=args.m, seed=args.seed)
    elif args.graph == "grid":
        mdp, _ = penv.grid_maze(height=args.height, width=args.width, wall_prob=args.wall_prob, seed=args.seed)
    else:
        raise ValueError("Unknown graph type")

    env = penv.GraphEnv(mdp)
    if args.shaping:
        if args.potential == "distance":
            pot = DistancePotential(mdp)
        else:
            pot = HarmonicPotential(mdp)
        env = RewardShaper(env, pot, gamma=0.99)
    return env


def main():
    args = parse_args()
    env = make_env(args)
    agent = TabularQAgent(env.observation_space.n, env.action_space.n)
    trainer = Trainer(env=env, agent=agent, train_episodes=args.episodes, seed=args.seed)
    summary = trainer.run()
    print("Training summary:", summary)


if __name__ == "__main__":
    main() 