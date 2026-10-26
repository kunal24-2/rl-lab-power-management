"""
Baseline policies for benchmarking RL-based Computer Lab Power Management.

Includes static single-mode policies (Always-Active, Always-Idle, Always-Sleep)
and a stochastic Random policy, all exposing a unified `select_action(state_idx)`
interface compatible with the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from lab_env import ComputerLabEnv


class BasePolicy(ABC):
    """Abstract base class for all power management policies."""

    name: str = "BasePolicy"

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset any internal policy state at the start of an episode."""
        pass

    @abstractmethod
    def select_action(self, state_idx: int) -> int:
        """Select a discrete action in {0: SLEEP, 1: IDLE, 2: ACTIVE}."""
        raise NotImplementedError


class AlwaysActivePolicy(BasePolicy):
    """Static baseline that keeps all standby computers in ACTIVE mode (action = 2)."""

    name: str = "Always-Active"

    def select_action(self, state_idx: int) -> int:
        return ComputerLabEnv.MODE_ACTIVE


class AlwaysIdlePolicy(BasePolicy):
    """Static baseline that keeps all standby computers in IDLE mode (action = 1)."""

    name: str = "Always-Idle"

    def select_action(self, state_idx: int) -> int:
        return ComputerLabEnv.MODE_IDLE


class AlwaysSleepPolicy(BasePolicy):
    """Static baseline that keeps all standby computers in SLEEP mode (action = 0)."""

    name: str = "Always-Sleep"

    def select_action(self, state_idx: int) -> int:
        return ComputerLabEnv.MODE_SLEEP


class RandomPolicy(BasePolicy):
    """Stochastic baseline that uniformly samples an action in {0, 1, 2} at each step."""

    name: str = "Random"

    def __init__(self, num_actions: int = 3, seed: Optional[int] = 42) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select_action(self, state_idx: int) -> int:
        return int(self.rng.integers(0, self.num_actions))


def get_baseline_policies(seed: int = 42) -> dict:
    """Return a dictionary of all standard baseline policy instances."""
    return {
        "Always-Active": AlwaysActivePolicy(),
        "Always-Idle": AlwaysIdlePolicy(),
        "Always-Sleep": AlwaysSleepPolicy(),
        "Random": RandomPolicy(seed=seed),
    }
