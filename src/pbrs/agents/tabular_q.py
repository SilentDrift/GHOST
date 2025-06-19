from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class TabularQAgent:
    """Simple tabular Q-learning agent for discrete state/action spaces."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or np.random.default_rng()

        # Q-table initialised to optimistic or zeros.
        self.Q = np.full((n_states, n_actions), fill_value=optimistic_init, dtype=float)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # Greedy tie-breaking
        return int(self.rng.choice(np.flatnonzero(self.Q[state] == self.Q[state].max())))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        """One-step Q-learning update."""
        target = r
        if not done:
            target += self.gamma * self.Q[s_next].max()
        td = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        s, _ = env.reset()
        episode_return = 0.0
        steps = 0
        success = False

        def _underlying_mdp(environment):
            base = environment
            while hasattr(base, 'env'):
                base = base.env
            return getattr(base, 'mdp', None)

        mdp_ref = _underlying_mdp(env)

        for t in range(max_steps):
            # Determine valid actions: outgoing edges from current node
            if mdp_ref is not None:
                valid_actions = list(mdp_ref.G.successors(s))
            else:
                valid_actions = list(range(self.n_actions))
            if not valid_actions:
                # Dead-end; end episode as failure
                break

            if self.rng.random() < self.epsilon:
                a = int(self.rng.choice(valid_actions))
            else:
                # Greedy over valid actions
                q_vals = self.Q[s, valid_actions]
                a = int(self.rng.choice([act for act, q in zip(valid_actions, q_vals) if q == q_vals.max()]))

            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            self.update(s, a, r, s_next, done)
            episode_return += r
            s = s_next
            steps += 1
            if done:
                success = bool(terminated)  # success if reached goal (terminated)
                break
        self.decay_epsilon()
        return {"return": episode_return, "steps": steps, "success": success} 