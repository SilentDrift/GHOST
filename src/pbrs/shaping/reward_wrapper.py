from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym

from ..potentials.base import Potential

__all__ = ["RewardShaper"]


class RewardShaper(gym.Wrapper):
    """Gym wrapper that adds potential-based shaping reward.

    Shaping term: F = γ Φ(s') - Φ(s)
    Requires Φ(goal)=Φ(fail)=0 to preserve policy invariance.
    """

    def __init__(self, env: gym.Env, potential: Potential, gamma: float):
        super().__init__(env)
        self.potential = potential
        self.gamma = gamma

        # Sanity check: terminal states should have zero potential
        if hasattr(env, "mdp"):
            mdp = env.mdp  # type: ignore[attr-defined]
            for t in mdp.goal_nodes + mdp.fail_nodes:
                assert abs(potential(t)) < 1e-8, "Potential at terminal must be 0"

        self._prev_state: int | None = None

    def reset(self, *args, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(*args, **kwargs)
        self._prev_state = obs
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert self._prev_state is not None, "reset() not called"
        F = self.gamma * self.potential(obs) - self.potential(self._prev_state)
        shaped_r = reward + F
        info = dict(info)
        info["shaping"] = F
        self._prev_state = obs
        return obs, shaped_r, terminated, truncated, info 