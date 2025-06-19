from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from pbrs.agents import TabularQAgent
from pbrs.envs.wrappers import GraphEnv
from pbrs.utils.seed import set_global_seed


@dataclass
class Trainer:
    env: GraphEnv
    agent: TabularQAgent
    train_episodes: int = 1000
    max_steps_per_episode: int = 1000
    seed: int = 0
    log_every: int = 100

    returns: List[float] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)

    def run(self) -> Dict[str, Any]:
        set_global_seed(self.seed)
        for ep in range(1, self.train_episodes + 1):
            stats = self.agent.train_episode(self.env, self.max_steps_per_episode)
            self.returns.append(stats["return"])
            self.successes.append(stats["success"])
            if self.log_every and ep % self.log_every == 0:
                mean_ret = sum(self.returns[-self.log_every:]) / self.log_every
                succ_rate = sum(self.successes[-self.log_every:]) / self.log_every
                print(f"Episode {ep}: mean_return={mean_ret:.3f}, success_rate={succ_rate:.2%}")
        summary = {
            "episodes": self.train_episodes,
            "mean_return": sum(self.returns) / self.train_episodes,
            "success_rate": sum(self.successes) / self.train_episodes,
        }
        return summary 