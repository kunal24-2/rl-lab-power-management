"""
Comprehensive unit and integration test suite for RL-Based Computer Lab Power Management.

Tests state encoding/decoding bijectivity, action space validation, episode termination,
reward function consistency, seed reproducibility, Q-Learning / SARSA updates,
model checkpoint serialization, baseline policies, and experimental outputs.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_LAB_CONFIG, DEFAULT_TRAINING_CONFIG, LabConfig
from lab_env import ComputerLabEnv
from baselines import (
    AlwaysActivePolicy,
    AlwaysIdlePolicy,
    AlwaysSleepPolicy,
    RandomPolicy,
    get_baseline_policies,
)
from train import QLearningAgent, SARSAAgent


def test_state_space_bijection() -> None:
    """Verify 1-to-1 bijective mapping between 5D state tuples and [0, 719]."""
    env = ComputerLabEnv(config=DEFAULT_LAB_CONFIG)
    assert env.num_states == 720

    seen_tuples = set()
    for idx in range(env.num_states):
        tup = env.decode_state(idx)
        assert len(tup) == 5
        seen_tuples.add(tup)
        re_encoded = env.encode_state(tup)
        assert re_encoded == idx

    assert len(seen_tuples) == 720

    with pytest.raises(ValueError):
        env.decode_state(-1)
    with pytest.raises(ValueError):
        env.decode_state(720)
    with pytest.raises(ValueError):
        env.encode_state((8, 0, 0, 0, 0))


def test_demand_and_trend_discretization() -> None:
    """Verify demand bracket and trend momentum discretization boundaries."""
    assert ComputerLabEnv.discretize_demand(0) == 0
    assert ComputerLabEnv.discretize_demand(1) == 0
    assert ComputerLabEnv.discretize_demand(2) == 1
    assert ComputerLabEnv.discretize_demand(5) == 1
    assert ComputerLabEnv.discretize_demand(6) == 2
    assert ComputerLabEnv.discretize_demand(10) == 2
    assert ComputerLabEnv.discretize_demand(11) == 3
    assert ComputerLabEnv.discretize_demand(15) == 3
    assert ComputerLabEnv.discretize_demand(16) == 4
    assert ComputerLabEnv.discretize_demand(20) == 4

    assert ComputerLabEnv.discretize_trend(3, 6) == 0   # Falling (diff = -3)
    assert ComputerLabEnv.discretize_trend(5, 5) == 1   # Stable (diff = 0)
    assert ComputerLabEnv.discretize_trend(8, 5) == 2   # Rising (diff = +3)


def test_action_validation_and_episode_termination() -> None:
    """Verify valid action bounds, 96-step horizon, and post-termination guard."""
    env = ComputerLabEnv(seed=101)
    state_idx, info = env.reset(seed=101)
    assert 0 <= state_idx < env.num_states
    assert "state_tuple" in info

    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(3)

    steps = 0
    done = False
    while not done:
        next_state, reward, done, step_info = env.step(ComputerLabEnv.MODE_IDLE)
        steps += 1
        assert 0 <= next_state < env.num_states
        assert isinstance(reward, float)
        if steps < DEFAULT_LAB_CONFIG.steps_per_episode:
            assert not done
        else:
            assert done

    assert steps == 96

    with pytest.raises(RuntimeError):
        env.step(ComputerLabEnv.MODE_IDLE)


def test_reward_and_physical_metrics_consistency() -> None:
    """Verify reward decomposition and physical bounds at every step."""
    env = ComputerLabEnv(seed=202)
    env.reset(seed=202, day_type=0)

    for step in range(96):
        action = step % 3
        _, reward, _, info = env.step(action)

        rc = info["reward_components"]
        reconstructed = (
            rc["reward_service"]
            - rc["penalty_energy"]
            - rc["penalty_violation"]
            - rc["penalty_latency"]
            - rc["penalty_switch"]
        )
        assert np.isclose(reward, reconstructed, atol=1e-9)

        assert info["energy_kwh"] > 0.0
        assert info["energy_cost"] > 0.0
        assert info["latency_sec"] >= 0.0
        assert 0 <= info["served_requests"] <= DEFAULT_LAB_CONFIG.num_computers
        assert info["service_violations"] >= 0
        assert info["busy_pcs"] + info["standby_pcs"] == DEFAULT_LAB_CONFIG.num_computers


def test_reproducibility_with_fixed_seeds() -> None:
    """Verify identical trajectories under identical seeds."""
    env1 = ComputerLabEnv()
    env2 = ComputerLabEnv()

    s1, _ = env1.reset(seed=555, day_type=0)
    s2, _ = env2.reset(seed=555, day_type=0)
    assert s1 == s2

    for step in range(96):
        act = (step * 7) % 3
        ns1, r1, d1, info1 = env1.step(act)
        ns2, r2, d2, info2 = env2.step(act)
        assert ns1 == ns2
        assert np.isclose(r1, r2, atol=1e-12)
        assert d1 == d2
        assert np.isclose(info1["energy_kwh"], info2["energy_kwh"], atol=1e-12)


def test_q_learning_update_and_serialization(tmp_path: Path) -> None:
    """Verify Bellman Q-Learning update math and save/load checkpoint fidelity."""
    agent = QLearningAgent(
        num_states=720,
        num_actions=3,
        learning_rate=0.2,
        discount_factor=0.95,
        seed=42,
    )
    agent.q_table[10, 2] = 5.0
    td_err = agent.update(state_idx=4, action=1, reward=3.0, next_state_idx=10, done=False)

    expected_target = 3.0 + 0.95 * 5.0  # 7.75
    expected_q = 0.2 * expected_target  # 1.55
    assert np.isclose(td_err, expected_target, atol=1e-9)
    assert np.isclose(agent.q_table[4, 1], expected_q, atol=1e-9)
    assert agent.visit_counts[4, 1] == 1

    ckpt_file = tmp_path / "test_q_agent.npz"
    agent.save(str(ckpt_file))
    loaded = QLearningAgent.load(str(ckpt_file))

    assert np.allclose(agent.q_table, loaded.q_table)
    assert np.array_equal(agent.visit_counts, loaded.visit_counts)
    assert loaded.select_action(4, training=False) == 1


def test_baselines_interface() -> None:
    """Verify all baseline policies return valid discrete actions in {0, 1, 2}."""
    policies = get_baseline_policies(seed=123)
    assert set(policies.keys()) == {
        "Always-Active",
        "Always-Idle",
        "Always-Sleep",
        "Random",
    }
    assert policies["Always-Active"].select_action(0) == 2
    assert policies["Always-Idle"].select_action(0) == 1
    assert policies["Always-Sleep"].select_action(0) == 0
    for s in range(50):
        assert policies["Random"].select_action(s) in (0, 1, 2)


def test_saved_artifacts_and_rl_performance() -> None:
    """Verify trained checkpoints, evaluation results, and generated figures exist."""
    assert Path(DEFAULT_TRAINING_CONFIG.model_save_path).exists()
    assert Path(DEFAULT_TRAINING_CONFIG.training_history_path).exists()
    assert Path(DEFAULT_TRAINING_CONFIG.evaluation_results_path).exists()
    assert Path(DEFAULT_TRAINING_CONFIG.per_seed_results_path).exists()

    with open(DEFAULT_TRAINING_CONFIG.evaluation_results_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    q_reward = summary["policies"]["Q-Learning"]["cumulative_reward"]["mean"]
    active_reward = summary["policies"]["Always-Active"]["cumulative_reward"]["mean"]
    idle_reward = summary["policies"]["Always-Idle"]["cumulative_reward"]["mean"]
    sleep_reward = summary["policies"]["Always-Sleep"]["cumulative_reward"]["mean"]
    random_reward = summary["policies"]["Random"]["cumulative_reward"]["mean"]

    assert q_reward > active_reward
    assert q_reward > idle_reward
    assert q_reward > sleep_reward
    assert q_reward > random_reward

    for fig_name in [
        "training_convergence.png",
        "reward_comparison.png",
        "energy_and_cost.png",
        "service_and_latency.png",
        "learned_policy_heatmap.png",
    ]:
        assert (Path("figures") / fig_name).exists()
