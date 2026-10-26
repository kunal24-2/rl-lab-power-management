"""
Configuration module for RL-Based University Computer Lab Power Management.

Defines physical parameters, environment constants, Markov Decision Process (MDP)
dimensions, reward weights, and training/evaluation hyperparameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LabConfig:
    """Physical and operational parameters of the university computer lab."""

    # Lab capacity and episode horizon
    num_computers: int = 20
    steps_per_episode: int = 96          # 24 hours * 4 steps/hour (15-min intervals)
    step_duration_hours: float = 0.25    # 15 minutes = 0.25 hours

    # Discrete state space dimensions: s_t = (period, day_type, demand_bin, trend_bin, mode)
    num_time_periods: int = 8            # 8 three-hour diurnal blocks across 24h (0..7)
    num_day_types: int = 2               # 0 = Weekday, 1 = Weekend
    num_demand_bins: int = 5             # 0: Very Low, 1: Low, 2: Moderate, 3: High, 4: Peak
    num_trend_bins: int = 3              # 0: Falling, 1: Stable, 2: Rising
    num_power_modes: int = 3             # 0: SLEEP, 1: IDLE, 2: ACTIVE

    # Action space definitions
    action_names: Tuple[str, ...] = ("SLEEP", "IDLE", "ACTIVE")

    # Power consumption per computer (in Watts)
    power_busy_watts: float = 150.0            # Computer actively running a student session
    power_active_standby_watts: float = 110.0  # Unoccupied PC kept in ACTIVE ready state
    power_idle_standby_watts: float = 45.0     # Unoccupied PC in warm IDLE (display off, C-state)
    power_sleep_standby_watts: float = 8.0     # Unoccupied PC in deep SLEEP (Suspend-to-RAM S3)

    # Mode transition energy overhead per switched standby computer (in kWh)
    switch_energy_per_pc_kwh: float = 0.0035

    # Wake-up / response latency parameters (in seconds)
    latency_active_sec: float = 1.5            # Instant login when standby pool is ACTIVE
    latency_idle_sec: float = 5.5              # Wake from warm IDLE
    latency_sleep_sec: float = 45.0            # Cold/S3 resume from deep SLEEP
    latency_mode_transition_sec: float = 4.0   # Additional delay when waking pool from SLEEP

    # Reference latency for reward normalization (in seconds)
    latency_ref_sec: float = 10.0

    # Time-of-Use (ToU) electricity tariff (USD per kWh)
    # Off-peak (00:00-05:59, 21:00-23:59): $0.08/kWh
    # Standard (06:00-11:59, 18:00-20:59): $0.12/kWh
    # Peak (12:00-17:59): $0.18/kWh
    tariff_offpeak_per_kwh: float = 0.08
    tariff_standard_per_kwh: float = 0.12
    tariff_peak_per_kwh: float = 0.18

    # Multi-objective reward function weights
    weight_service: float = 1.25         # Positive reward per served student session
    weight_energy: float = 6.00          # Penalty per kWh of electrical energy consumed
    weight_violation: float = 6.50       # Penalty per unserved/dropped request (QoS violation)
    weight_latency: float = 1.50         # Penalty per normalized latency unit (L_t / L_ref)
    weight_switch: float = 0.70          # Penalty for switching standby power mode (a_t != m_t)

    @property
    def num_states(self) -> int:
        """Total cardinality of the discrete state space |S| = 8 * 2 * 5 * 3 * 3 = 720."""
        return (
            self.num_time_periods
            * self.num_day_types
            * self.num_demand_bins
            * self.num_trend_bins
            * self.num_power_modes
        )

    @property
    def num_actions(self) -> int:
        """Total cardinality of the discrete action space |A| = 3."""
        return len(self.action_names)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for RL training and evaluation."""

    # Training settings
    num_episodes: int = 3000
    learning_rate: float = 0.15
    min_learning_rate: float = 0.015
    lr_decay: float = 0.9992
    discount_factor: float = 0.96        # Gamma

    # Exploration schedule (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9982

    # Reproducibility seeds (strictly disjoint between train and eval)
    train_seed: int = 42
    eval_base_seed: int = 1000
    num_eval_seeds: int = 20
    episodes_per_eval_seed: int = 5      # 20 seeds * 5 episodes = 100 evaluation episodes

    # Artifact file paths
    model_save_path: str = "models/q_learning_agent.npz"
    sarsa_model_save_path: str = "models/sarsa_agent.npz"
    training_history_path: str = "results/training_history.json"
    evaluation_results_path: str = "results/evaluation_summary.json"
    per_seed_results_path: str = "results/evaluation_per_seed.csv"


DEFAULT_LAB_CONFIG = LabConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
