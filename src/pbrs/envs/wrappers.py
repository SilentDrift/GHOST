from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces

from .graph_env import GraphMDP

__all__ = ["GraphEnv"]


class GraphEnv(gym.Env):
    """Adapter that exposes :class:`GraphMDP` through the Gymnasium API."""

    metadata = {
        "render_modes": [None],
    }

    def __init__(self, mdp: GraphMDP):
        super().__init__()
        self.mdp: GraphMDP = mdp
        n_states = self.mdp.n_states
        # Observations are state indices in [0, n_states).
        self.observation_space = spaces.Discrete(n_states)
        # We allow any target node as an *action*; invalid choices will raise.
        self.action_space = spaces.Discrete(n_states)

        self._state: int | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Reset the underlying MDP and return the first observation.

        Args:
            seed: Optional seed to re-seed the wrapped MDP for this call.
            options: Optional ``dict`` containing extra arguments. If provided,
                ``options["start"]`` specifies the start node and must be a
                non-terminal state.

        Returns:
            ``(observation, info)`` according to the Gymnasium API.
        """
        super().reset(seed=seed)
        start = options.get("start") if options else None
        self._state = self.mdp.reset(seed=seed, start=start)
        observation = int(self._state)
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        next_state, reward, terminated, info = self.mdp.step(action)
        observation = int(next_state)
        # No truncation logic by default.
        truncated = False
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def render(self):
        # For now, we don't implement a visualiser. Could print state.
        print(f"Current state: {self.mdp.state}")

    def close(self):
        pass 