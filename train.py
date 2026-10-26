"""
Reinforcement Learning agent implementation and training pipeline (train.py).

Implements Tabular Q-Learning (off-policy TD control) and Tabular SARSA
(on-policy TD control) with epsilon-greedy exploration, learning rate scheduling,
reproducible seeding, and checkpoint serialization.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from baselines import BasePolicy
from config import (
    DEFAULT_LAB_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    LabConfig,
    TrainingConfig,
)
from lab_env import ComputerLabEnv


class QLearningAgent(BasePolicy):
    """
    Tabular Off-Policy Q-Learning Agent.

    Bellman Optimality TD(0) Update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
    """

    name: str = "Q-Learning"

    def __init__(
        self,
        num_states: int = DEFAULT_LAB_CONFIG.num_states,
        num_actions: int = DEFAULT_LAB_CONFIG.num_actions,
        learning_rate: float = DEFAULT_TRAINING_CONFIG.learning_rate,
        min_learning_rate: float = DEFAULT_TRAINING_CONFIG.min_learning_rate,
        lr_decay: float = DEFAULT_TRAINING_CONFIG.lr_decay,
        discount_factor: float = DEFAULT_TRAINING_CONFIG.discount_factor,
        epsilon_start: float = DEFAULT_TRAINING_CONFIG.epsilon_start,
        epsilon_min: float = DEFAULT_TRAINING_CONFIG.epsilon_min,
        epsilon_decay: float = DEFAULT_TRAINING_CONFIG.epsilon_decay,
        seed: Optional[int] = DEFAULT_TRAINING_CONFIG.train_seed,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_decay = lr_decay
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)
        self.visit_counts = np.zeros((num_states, num_actions), dtype=np.int64)
        self.training_episodes_completed = 0

    def select_action(self, state_idx: int, training: bool = False) -> int:
        """
        Select an action using epsilon-greedy exploration during training
        or greedy exploitation over visited actions during evaluation.
        """
        counts = self.visit_counts[state_idx]
        if training:
            # Ensure unvisited actions in a reached state are explored first
            unvisited = np.flatnonzero(counts == 0)
            if len(unvisited) > 0:
                return int(self.rng.choice(unvisited))
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(0, self.num_actions))

        q_values = self.q_table[state_idx]
        if not training and np.any(counts > 0):
            masked_q = np.where(counts > 0, q_values, -np.inf)
            max_q = np.max(masked_q)
            best_actions = np.flatnonzero(np.isclose(masked_q, max_q, atol=1e-9))
            return int(best_actions[0])

        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q, atol=1e-9))
        if len(best_actions) == 1 or not training:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """
        Perform a single Q-Learning TD(0) update and return the absolute TD error.
        """
        self.visit_counts[state_idx, action] += 1
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state_idx]))
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.learning_rate * td_error
        return float(abs(td_error))

    def decay_schedules(self) -> None:
        """Decay exploration rate epsilon and learning rate alpha after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.learning_rate = max(
            self.min_learning_rate, self.learning_rate * self.lr_decay
        )
        self.training_episodes_completed += 1

    def save(self, filepath: str) -> None:
        """Serialize the learned Q-table, visit counts, and hyperparameters to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            filepath,
            q_table=self.q_table,
            visit_counts=self.visit_counts,
            num_states=np.array(self.num_states),
            num_actions=np.array(self.num_actions),
            learning_rate=np.array(self.learning_rate),
            initial_lr=np.array(self.initial_lr),
            min_learning_rate=np.array(self.min_learning_rate),
            lr_decay=np.array(self.lr_decay),
            discount_factor=np.array(self.discount_factor),
            epsilon=np.array(self.epsilon),
            epsilon_start=np.array(self.epsilon_start),
            epsilon_min=np.array(self.epsilon_min),
            epsilon_decay=np.array(self.epsilon_decay),
            training_episodes_completed=np.array(self.training_episodes_completed),
        )

    @classmethod
    def load(cls, filepath: str, seed: Optional[int] = None) -> "QLearningAgent":
        """Load a saved Q-Learning checkpoint from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")
        data = np.load(filepath)
        agent = cls(
            num_states=int(data["num_states"]),
            num_actions=int(data["num_actions"]),
            learning_rate=float(data["learning_rate"]),
            min_learning_rate=float(data["min_learning_rate"]),
            lr_decay=float(data["lr_decay"]),
            discount_factor=float(data["discount_factor"]),
            epsilon_start=float(data["epsilon_start"]),
            epsilon_min=float(data["epsilon_min"]),
            epsilon_decay=float(data["epsilon_decay"]),
            seed=seed,
        )
        agent.q_table = data["q_table"].astype(np.float64)
        agent.visit_counts = data["visit_counts"].astype(np.int64)
        agent.epsilon = float(data["epsilon"])
        agent.training_episodes_completed = int(data["training_episodes_completed"])
        return agent


class SARSAAgent(QLearningAgent):
    """
    Tabular On-Policy SARSA Agent for comparative RL benchmarking.

    On-Policy TD(0) Update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    """

    name: str = "SARSA"

    def update_sarsa(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform an on-policy SARSA TD(0) update and return the absolute TD error."""
        self.visit_counts[state_idx, action] += 1
        next_q = 0.0 if done else float(self.q_table[next_state_idx, next_action])
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.learning_rate * td_error
        return float(abs(td_error))


def train_q_learning(
    lab_config: LabConfig = DEFAULT_LAB_CONFIG,
    train_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    verbose: bool = True,
) -> Tuple[QLearningAgent, Dict[str, List[float]]]:
    """
    Train the Tabular Q-Learning agent on the ComputerLabEnv environment.

    Uses a dedicated training random seed strictly separated from the held-out
    evaluation seeds to prevent any training-evaluation data leakage.
    """
    env = ComputerLabEnv(config=lab_config, seed=train_config.train_seed)
    agent = QLearningAgent(
        num_states=lab_config.num_states,
        num_actions=lab_config.num_actions,
        learning_rate=train_config.learning_rate,
        min_learning_rate=train_config.min_learning_rate,
        lr_decay=train_config.lr_decay,
        discount_factor=train_config.discount_factor,
        epsilon_start=train_config.epsilon_start,
        epsilon_min=train_config.epsilon_min,
        epsilon_decay=train_config.epsilon_decay,
        seed=train_config.train_seed,
    )

    history: Dict[str, List[float]] = {
        "episode": [],
        "episode_reward": [],
        "energy_kwh": [],
        "energy_cost": [],
        "served_requests": [],
        "service_violations": [],
        "mean_latency_sec": [],
        "mode_switches": [],
        "epsilon": [],
        "learning_rate": [],
        "mean_td_error": [],
    }

    for ep in range(1, train_config.num_episodes + 1):
        # Deterministic per-episode training seed derived from base train_seed
        ep_seed = train_config.train_seed * 10000 + ep
        state_idx, _ = env.reset(seed=ep_seed)

        ep_reward = 0.0
        ep_energy = 0.0
        ep_cost = 0.0
        ep_served = 0
        ep_violations = 0
        ep_switches = 0
        latencies: List[float] = []
        td_errors: List[float] = []

        done = False
        while not done:
            action = agent.select_action(state_idx, training=True)
            next_state_idx, reward, done, info = env.step(action)
            td_err = agent.update(state_idx, action, reward, next_state_idx, done)

            ep_reward += reward
            ep_energy += info["energy_kwh"]
            ep_cost += info["energy_cost"]
            ep_served += info["served_requests"]
            ep_violations += info["service_violations"]
            ep_switches += info["mode_switched"]
            latencies.append(info["latency_sec"])
            td_errors.append(td_err)

            state_idx = next_state_idx

        history["episode"].append(ep)
        history["episode_reward"].append(float(ep_reward))
        history["energy_kwh"].append(float(ep_energy))
        history["energy_cost"].append(float(ep_cost))
        history["served_requests"].append(int(ep_served))
        history["service_violations"].append(int(ep_violations))
        history["mean_latency_sec"].append(float(np.mean(latencies)))
        history["mode_switches"].append(int(ep_switches))
        history["epsilon"].append(float(agent.epsilon))
        history["learning_rate"].append(float(agent.learning_rate))
        history["mean_td_error"].append(float(np.mean(td_errors)))

        agent.decay_schedules()

        if verbose and (ep == 1 or ep % 500 == 0 or ep == train_config.num_episodes):
            recent_r = np.mean(history["episode_reward"][-100:])
            recent_e = np.mean(history["energy_kwh"][-100:])
            recent_v = np.mean(history["service_violations"][-100:])
            print(
                f"[Q-Learning] Episode {ep:4d}/{train_config.num_episodes} | "
                f"Avg100 Reward: {recent_r:7.2f} | "
                f"Energy: {recent_e:5.2f} kWh | "
                f"Violations: {recent_v:4.2f} | "
                f"Eps: {agent.epsilon:.3f}"
            )

    agent.save(train_config.model_save_path)
    return agent, history


def train_sarsa(
    lab_config: LabConfig = DEFAULT_LAB_CONFIG,
    train_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    verbose: bool = True,
) -> Tuple[SARSAAgent, Dict[str, List[float]]]:
    """
    Train the On-Policy Tabular SARSA agent under the exact same training schedule.
    """
    env = ComputerLabEnv(config=lab_config, seed=train_config.train_seed)
    agent = SARSAAgent(
        num_states=lab_config.num_states,
        num_actions=lab_config.num_actions,
        learning_rate=train_config.learning_rate,
        min_learning_rate=train_config.min_learning_rate,
        lr_decay=train_config.lr_decay,
        discount_factor=train_config.discount_factor,
        epsilon_start=train_config.epsilon_start,
        epsilon_min=train_config.epsilon_min,
        epsilon_decay=train_config.epsilon_decay,
        seed=train_config.train_seed,
    )

    history: Dict[str, List[float]] = {
        "episode": [],
        "episode_reward": [],
        "energy_kwh": [],
        "service_violations": [],
    }

    for ep in range(1, train_config.num_episodes + 1):
        ep_seed = train_config.train_seed * 10000 + ep
        state_idx, _ = env.reset(seed=ep_seed)
        action = agent.select_action(state_idx, training=True)

        ep_reward = 0.0
        ep_energy = 0.0
        ep_violations = 0
        done = False

        while not done:
            next_state_idx, reward, done, info = env.step(action)
            next_action = (
                0 if done else agent.select_action(next_state_idx, training=True)
            )
            agent.update_sarsa(
                state_idx, action, reward, next_state_idx, next_action, done
            )

            ep_reward += reward
            ep_energy += info["energy_kwh"]
            ep_violations += info["service_violations"]

            state_idx = next_state_idx
            action = next_action

        history["episode"].append(ep)
        history["episode_reward"].append(float(ep_reward))
        history["energy_kwh"].append(float(ep_energy))
        history["service_violations"].append(int(ep_violations))
        agent.decay_schedules()

        if verbose and (ep == 1 or ep % 500 == 0 or ep == train_config.num_episodes):
            recent_r = np.mean(history["episode_reward"][-100:])
            print(
                f"[SARSA]      Episode {ep:4d}/{train_config.num_episodes} | "
                f"Avg100 Reward: {recent_r:7.2f} | Eps: {agent.epsilon:.3f}"
            )

    agent.save(train_config.sarsa_model_save_path)
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL agents for University Computer Lab Power Management."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_episodes,
        help="Number of training episodes (default: 3000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.train_seed,
        help="Training random seed (default: 42)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.learning_rate,
        help="Initial learning rate alpha (default: 0.15)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.discount_factor,
        help="Discount factor gamma (default: 0.96)",
    )
    args = parser.parse_args()

    train_cfg = TrainingConfig(
        num_episodes=args.episodes,
        train_seed=args.seed,
        learning_rate=args.lr,
        discount_factor=args.gamma,
    )

    print("=" * 72)
    print("Training RL Agents for Computer Lab Power Management")
    print(
        f"State Space |S| = {DEFAULT_LAB_CONFIG.num_states} | "
        f"Action Space |A| = {DEFAULT_LAB_CONFIG.num_actions} | "
        f"Episodes = {train_cfg.num_episodes}"
    )
    print("=" * 72)

    q_agent, q_history = train_q_learning(DEFAULT_LAB_CONFIG, train_cfg, verbose=True)
    print("-" * 72)
    sarsa_agent, sarsa_history = train_sarsa(DEFAULT_LAB_CONFIG, train_cfg, verbose=True)

    visited_states = int(np.sum(np.any(q_agent.visit_counts > 0, axis=1)))
    total_updates = int(np.sum(q_agent.visit_counts))
    print("-" * 72)
    print(f"Saved Q-Learning checkpoint to: {train_cfg.model_save_path}")
    print(f"Saved SARSA checkpoint to     : {train_cfg.sarsa_model_save_path}")
    print(
        f"State-space coverage: {visited_states}/{DEFAULT_LAB_CONFIG.num_states} "
        f"reachable states visited ({total_updates:,} total TD updates)"
    )

    # Save combined training history JSON
    Path(train_cfg.training_history_path).parent.mkdir(parents=True, exist_ok=True)
    combined_history = {
        "config": {
            "num_episodes": train_cfg.num_episodes,
            "learning_rate": train_cfg.learning_rate,
            "min_learning_rate": train_cfg.min_learning_rate,
            "lr_decay": train_cfg.lr_decay,
            "discount_factor": train_cfg.discount_factor,
            "epsilon_start": train_cfg.epsilon_start,
            "epsilon_min": train_cfg.epsilon_min,
            "epsilon_decay": train_cfg.epsilon_decay,
            "train_seed": train_cfg.train_seed,
            "visited_states": visited_states,
            "total_updates": total_updates,
        },
        "q_learning": q_history,
        "sarsa": sarsa_history,
    }
    with open(train_cfg.training_history_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f"Saved training history to     : {train_cfg.training_history_path}")


if __name__ == "__main__":
    main()
