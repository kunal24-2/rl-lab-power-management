"""
University Computer Lab Power Management Environment (lab_env.py).

Models a 20-workstation university computer lab operating over a 24-hour cycle
in 15-minute discrete time steps (T = 96 steps per episode). At each step, the
RL agent selects a target standby power mode in {0: SLEEP, 1: IDLE, 2: ACTIVE}
to balance electrical energy consumption, Time-of-Use (ToU) electricity cost,
mode-switching overhead, wake-up latency, and student Quality of Service (QoS).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from config import DEFAULT_LAB_CONFIG, LabConfig


class ComputerLabEnv:
    """
    Markov Decision Process (MDP) environment for Computer Lab Power Management.

    State Tuple s_t = (period, day_type, demand_bin, trend_bin, current_mode):
        - period (p_t)       : int in {0, ..., 7} (3-hour diurnal block: hour // 3)
        - day_type (d_t)     : int in {0, 1} (0 = Weekday, 1 = Weekend)
        - demand_bin (u_t)   : int in {0, 1, 2, 3, 4} (discretized active user count)
        - trend_bin (delta_t): int in {0, 1, 2} (0 = Falling, 1 = Stable, 2 = Rising)
        - current_mode (m_t) : int in {0, 1, 2} (0 = SLEEP, 1 = IDLE, 2 = ACTIVE)

    State space cardinality: |S| = 8 * 2 * 5 * 3 * 3 = 720 discrete states.
    Action space cardinality: |A| = 3 discrete actions {0: SLEEP, 1: IDLE, 2: ACTIVE}.
    """

    MODE_SLEEP = 0
    MODE_IDLE = 1
    MODE_ACTIVE = 2

    PERIOD_LABELS = (
        "00:00-02:59",
        "03:00-05:59",
        "06:00-08:59",
        "09:00-11:59",
        "12:00-14:59",
        "15:00-17:59",
        "18:00-20:59",
        "21:00-23:59",
    )

    def __init__(
        self,
        config: LabConfig = DEFAULT_LAB_CONFIG,
        seed: Optional[int] = None,
        fixed_day_type: Optional[int] = None,
    ) -> None:
        self.config = config
        self.fixed_day_type = fixed_day_type
        self.rng = np.random.default_rng(seed)

        # Dimensionality metadata
        self.state_shape: Tuple[int, int, int, int, int] = (
            self.config.num_time_periods,
            self.config.num_day_types,
            self.config.num_demand_bins,
            self.config.num_trend_bins,
            self.config.num_power_modes,
        )
        self.num_states: int = self.config.num_states
        self.num_actions: int = self.config.num_actions

        # Internal episode state variables
        self.step_idx: int = 0
        self.day_type: int = 0
        self.active_users: int = 0
        self.prev_active_users: int = 0
        self.current_mode: int = self.MODE_IDLE
        self.done: bool = False

    def seed(self, seed: Optional[int] = None) -> None:
        """Reset the environment's random number generator with a given seed."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # State Encoding & Decoding Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def discretize_demand(num_users: int) -> int:
        """
        Map raw active user count (0..20) to 5 discrete occupancy brackets:
            Bin 0 (Very Low) :  0 -  1 users
            Bin 1 (Low)      :  2 -  5 users
            Bin 2 (Moderate) :  6 - 10 users
            Bin 3 (High)     : 11 - 15 users
            Bin 4 (Peak)     : 16 - 20 users
        """
        if num_users <= 1:
            return 0
        if num_users <= 5:
            return 1
        if num_users <= 10:
            return 2
        if num_users <= 15:
            return 3
        return 4

    @staticmethod
    def discretize_trend(curr_users: int, prev_users: int) -> int:
        """
        Map step-over-step user count change to 3 discrete momentum bins:
            Bin 0 (Falling): diff <= -2
            Bin 1 (Stable) : -1 <= diff <= +1
            Bin 2 (Rising) : diff >= +2
        """
        diff = curr_users - prev_users
        if diff <= -2:
            return 0
        if diff >= 2:
            return 2
        return 1

    def get_state_tuple(self) -> Tuple[int, int, int, int, int]:
        """Return the current 5D discrete observation tuple s_t."""
        hour = (self.step_idx // 4) % 24
        period = hour // 3
        demand_bin = self.discretize_demand(self.active_users)
        trend_bin = self.discretize_trend(self.active_users, self.prev_active_users)
        return (period, self.day_type, demand_bin, trend_bin, self.current_mode)

    def encode_state(self, state_tuple: Tuple[int, int, int, int, int]) -> int:
        """Bijectively map a 5D state tuple to a flat integer index in [0, 719]."""
        period, day_type, demand_bin, trend_bin, mode = state_tuple
        if not (0 <= period < self.config.num_time_periods):
            raise ValueError(f"Invalid period: {period}")
        if not (0 <= day_type < self.config.num_day_types):
            raise ValueError(f"Invalid day_type: {day_type}")
        if not (0 <= demand_bin < self.config.num_demand_bins):
            raise ValueError(f"Invalid demand_bin: {demand_bin}")
        if not (0 <= trend_bin < self.config.num_trend_bins):
            raise ValueError(f"Invalid trend_bin: {trend_bin}")
        if not (0 <= mode < self.config.num_power_modes):
            raise ValueError(f"Invalid mode: {mode}")

        idx = period
        idx = idx * self.config.num_day_types + day_type
        idx = idx * self.config.num_demand_bins + demand_bin
        idx = idx * self.config.num_trend_bins + trend_bin
        idx = idx * self.config.num_power_modes + mode
        return int(idx)

    def decode_state(self, state_idx: int) -> Tuple[int, int, int, int, int]:
        """Bijectively decode a flat integer index in [0, 719] to a 5D state tuple."""
        if not (0 <= state_idx < self.num_states):
            raise ValueError(f"State index {state_idx} out of bounds [0, {self.num_states - 1}]")

        rem = int(state_idx)
        mode = rem % self.config.num_power_modes
        rem //= self.config.num_power_modes

        trend_bin = rem % self.config.num_trend_bins
        rem //= self.config.num_trend_bins

        demand_bin = rem % self.config.num_demand_bins
        rem //= self.config.num_demand_bins

        day_type = rem % self.config.num_day_types
        period = rem // self.config.num_day_types

        return (period, day_type, demand_bin, trend_bin, mode)

    # ------------------------------------------------------------------
    # Simulation Dynamics: Usage Patterns, Tariff, Power, QoS
    # ------------------------------------------------------------------

    def _expected_arrival_rate(self, hour: int, quarter: int, day_type: int) -> float:
        """
        Compute expected student arrivals in a 15-minute interval based on
        realistic university computer lab schedules.
        """
        if day_type == 0:  # Weekday schedule
            if 0 <= hour < 6:
                base = 0.04  # Overnight closed/empty lab
            elif 6 <= hour < 8:
                base = 0.25 if hour == 6 else 1.1  # Early morning ramp-up
            elif 8 <= hour < 12:
                # Scheduled morning practical lab sessions
                base = 4.4 if quarter in (0, 1) else 2.6
            elif 12 <= hour < 14:
                # Lunch break moderate usage
                base = 1.6
            elif 14 <= hour < 18:
                # Peak afternoon lab classes & assignment deadlines
                base = 5.2 if quarter in (0, 1) else 3.4
            elif 18 <= hour < 21:
                # Evening open self-study & coding practice
                base = 2.0
            elif 21 <= hour < 23:
                # Late evening wind-down
                base = 0.45
            else:
                base = 0.08
        else:  # Weekend schedule
            if 0 <= hour < 9:
                base = 0.05
            elif 9 <= hour < 13:
                base = 1.5
            elif 13 <= hour < 18:
                base = 2.4
            elif 18 <= hour < 21:
                base = 1.3
            else:
                base = 0.08

        return base

    def _departure_probability(self, hour: int, quarter: int, day_type: int) -> float:
        """Probability that an existing user ends their session in the current 15-min step."""
        if 0 <= hour < 7 or hour >= 21:
            return 0.60  # Fast turnover late at night
        if day_type == 0 and hour in (12, 17) and quarter >= 2:
            return 0.50  # End-of-lab-slot wave departures
        if day_type == 0 and (8 <= hour < 12 or 14 <= hour < 18):
            return 0.22  # Students stay seated during scheduled labs
        return 0.34

    def get_electricity_tariff(self, hour: int) -> float:
        """Return Time-of-Use (ToU) electricity rate ($/kWh) for a given hour."""
        if 12 <= hour < 18:
            return self.config.tariff_peak_per_kwh
        if (6 <= hour < 12) or (18 <= hour < 21):
            return self.config.tariff_standard_per_kwh
        return self.config.tariff_offpeak_per_kwh

    # ------------------------------------------------------------------
    # Gym-style Environment Interface: reset() and step()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        day_type: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment to the start of a 24-hour day (00:00).

        Returns:
            state_idx: Integer encoded initial state in [0, 719].
            info: Dictionary with initial state tuple and metadata.
        """
        if seed is not None:
            self.seed(seed)

        self.step_idx = 0
        if day_type is not None:
            self.day_type = int(day_type)
        elif self.fixed_day_type is not None:
            self.day_type = int(self.fixed_day_type)
        else:
            # 5/7 probability of Weekday (0), 2/7 probability of Weekend (1)
            self.day_type = 0 if self.rng.random() < (5.0 / 7.0) else 1

        # Initial overnight occupancy at 00:00
        self.active_users = 0
        self.prev_active_users = 0
        self.current_mode = self.MODE_IDLE
        self.done = False

        state_tuple = self.get_state_tuple()
        state_idx = self.encode_state(state_tuple)
        return state_idx, {
            "state_tuple": state_tuple,
            "active_users": self.active_users,
            "day_type": self.day_type,
        }

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one 15-minute time step under the selected power management action.

        Args:
            action: Discrete action in {0: SLEEP, 1: IDLE, 2: ACTIVE}.

        Returns:
            next_state_idx: Encoded next state in [0, 719].
            reward: Scalar step reward R_t.
            done: True if the 24-hour episode (96 steps) has finished.
            info: Detailed dictionary of step metrics and reward breakdown.
        """
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated episode. Call reset() first.")
        if action not in (self.MODE_SLEEP, self.MODE_IDLE, self.MODE_ACTIVE):
            raise ValueError(f"Invalid action {action}. Expected 0 (SLEEP), 1 (IDLE), or 2 (ACTIVE).")

        hour = (self.step_idx // 4) % 24
        quarter = self.step_idx % 4
        prev_mode = self.current_mode
        mode_switched = int(action != prev_mode)
        mode_jump = abs(action - prev_mode)

        # 1. Process departures from existing sessions
        dep_prob = self._departure_probability(hour, quarter, self.day_type)
        departures = int(self.rng.binomial(self.active_users, dep_prob))
        retained_users = self.active_users - departures

        # If aggressive SLEEP policy (action == 0) is applied while students are working,
        # brief idle pauses trigger unintended suspend events on ~20% of retained sessions.
        premature_suspends = 0
        if action == self.MODE_SLEEP and retained_users > 0:
            premature_suspends = int(self.rng.binomial(retained_users, 0.20))
            retained_users -= premature_suspends

        # 2. Sample new student arrivals during this 15-minute interval
        lam = self._expected_arrival_rate(hour, quarter, self.day_type)
        raw_arrivals = int(self.rng.poisson(lam))
        available_seats = max(0, self.config.num_computers - retained_users)
        arrivals = min(raw_arrivals, available_seats)

        # 3. Determine served arrivals, service violations, and wake-up latency
        #    based on the chosen power mode (action) and previous mode (prev_mode).
        if action == self.MODE_ACTIVE:
            served_new = arrivals
            arrival_violations = 0
            if arrivals == 0 and retained_users == 0:
                latency_sec = 0.0
            else:
                latency_sec = self.config.latency_active_sec
                if prev_mode == self.MODE_SLEEP and arrivals > 0:
                    latency_sec += self.config.latency_mode_transition_sec

        elif action == self.MODE_IDLE:
            # Warm IDLE handles moderate arrivals (up to 3 arrivals per 15-min slot)
            # without violations, whereas heavy lab-class arrival surges (>3) experience
            # wake-up queue congestion and login timeouts.
            warm_buffer = 3
            if arrivals <= warm_buffer:
                served_new = arrivals
                arrival_violations = 0
            else:
                surge = arrivals - warm_buffer
                arrival_violations = int(np.ceil(0.40 * surge))
                served_new = arrivals - arrival_violations

            if arrivals == 0 and retained_users == 0:
                latency_sec = 0.0
            elif arrivals == 0:
                # Only already-seated users are active; power-saver governor adds lag only at high load
                latency_sec = 1.8 if retained_users < 12 else 6.5
            else:
                congestion_delay = max(0, arrivals - 2) * 2.5
                high_load_throttle = 4.5 if (retained_users + served_new) >= 12 else 0.0
                transition_delay = 2.5 if (prev_mode == self.MODE_SLEEP) else 0.0
                latency_sec = (
                    self.config.latency_idle_sec
                    + congestion_delay
                    + high_load_throttle
                    + transition_delay
                )

        else:  # action == self.MODE_SLEEP
            # Deep SLEEP requires cold/S3 resume (45s) when arrivals occur;
            # when the lab is empty and no one arrives, zero users wait (0.0s latency).
            if arrivals == 0:
                served_new = 0
                arrival_violations = 0
                latency_sec = 0.0 if (retained_users == 0 and premature_suspends == 0) else 24.0
            else:
                arrival_violations = int(np.ceil(0.50 * arrivals))
                served_new = max(0, arrivals - arrival_violations)
                latency_sec = self.config.latency_sleep_sec + 2.0 * arrivals

        total_violations = premature_suspends + arrival_violations
        total_served = retained_users + served_new
        total_requests = retained_users + premature_suspends + arrivals

        # 4. Compute electrical energy consumption (kWh) and ToU electricity cost ($)
        busy_pcs = total_served
        standby_pcs = self.config.num_computers - busy_pcs

        if action == self.MODE_ACTIVE:
            standby_watts = self.config.power_active_standby_watts
        elif action == self.MODE_IDLE:
            standby_watts = self.config.power_idle_standby_watts
        else:
            standby_watts = self.config.power_sleep_standby_watts

        total_power_watts = (
            busy_pcs * self.config.power_busy_watts
            + standby_pcs * standby_watts
        )
        steady_energy_kwh = (total_power_watts / 1000.0) * self.config.step_duration_hours

        # Mode transition energy spike when switching standby pool power state
        switch_energy_kwh = 0.0
        if mode_switched and standby_pcs > 0:
            switch_energy_kwh = (
                standby_pcs
                * self.config.switch_energy_per_pc_kwh
                * (1.0 + 0.4 * (mode_jump - 1))
            )

        energy_kwh = steady_energy_kwh + switch_energy_kwh
        tariff = self.get_electricity_tariff(hour)
        energy_cost = energy_kwh * tariff

        # 5. Compute multi-objective reward R_t
        # R_t = w_srv * N_served - w_eng * E_t - w_viol * N_viol - w_lat * (L_t / L_ref) - w_sw * I(a_t != m_t)
        reward_service = self.config.weight_service * float(total_served)
        penalty_energy = self.config.weight_energy * float(energy_kwh)
        penalty_violation = self.config.weight_violation * float(total_violations)
        penalty_latency = self.config.weight_latency * (float(latency_sec) / self.config.latency_ref_sec)
        penalty_switch = self.config.weight_switch * float(mode_switched)

        reward = (
            reward_service
            - penalty_energy
            - penalty_violation
            - penalty_latency
            - penalty_switch
        )

        # 6. Advance environment state to t + 1
        self.prev_active_users = self.active_users
        self.active_users = total_served
        self.current_mode = int(action)
        self.step_idx += 1
        self.done = self.step_idx >= self.config.steps_per_episode

        next_state_tuple = self.get_state_tuple()
        next_state_idx = self.encode_state(next_state_tuple)

        info: Dict[str, Any] = {
            "step": self.step_idx,
            "hour": hour,
            "period": hour // 3,
            "quarter": quarter,
            "day_type": self.day_type,
            "action": int(action),
            "action_name": self.config.action_names[action],
            "prev_mode": prev_mode,
            "mode_switched": mode_switched,
            "arrivals": arrivals,
            "retained_users": retained_users,
            "served_requests": total_served,
            "total_requests": total_requests,
            "service_violations": total_violations,
            "premature_suspends": premature_suspends,
            "arrival_violations": arrival_violations,
            "busy_pcs": busy_pcs,
            "standby_pcs": standby_pcs,
            "power_watts": total_power_watts,
            "energy_kwh": energy_kwh,
            "tariff_per_kwh": tariff,
            "energy_cost": energy_cost,
            "latency_sec": latency_sec,
            "state_tuple": next_state_tuple,
            "reward_components": {
                "reward_service": reward_service,
                "penalty_energy": penalty_energy,
                "penalty_violation": penalty_violation,
                "penalty_latency": penalty_latency,
                "penalty_switch": penalty_switch,
            },
        }

        return next_state_idx, float(reward), self.done, info
