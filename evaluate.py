"""
Evaluation pipeline for RL-Based Computer Lab Power Management (evaluate.py).

Loads trained RL checkpoints (Tabular Q-Learning and SARSA) and benchmarks them
against baseline policies (Always-Active, Always-Idle, Always-Sleep, Random)
across multiple independent held-out evaluation seeds.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from baselines import BasePolicy, get_baseline_policies
from config import (
    DEFAULT_LAB_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    LabConfig,
    TrainingConfig,
)
from lab_env import ComputerLabEnv
from train import QLearningAgent, SARSAAgent


def evaluate_policy_on_episode(
    env: ComputerLabEnv,
    policy: BasePolicy,
    episode_seed: int,
    day_type: int,
) -> Dict[str, Any]:
    """Run a single 24-hour evaluation episode (96 steps) and collect all metrics."""
    policy.reset(seed=episode_seed + 777)
    state_idx, _ = env.reset(seed=episode_seed, day_type=day_type)

    cum_reward = 0.0
    total_energy_kwh = 0.0
    total_energy_cost = 0.0
    total_served = 0
    total_requests = 0
    total_violations = 0
    total_switches = 0
    latencies: List[float] = []
    action_counts = [0, 0, 0]

    hourly_power_kw = np.zeros(24, dtype=np.float64)
    hourly_users = np.zeros(24, dtype=np.float64)
    hourly_actions = np.zeros(24, dtype=np.float64)

    done = False
    while not done:
        action = policy.select_action(state_idx)
        next_state_idx, reward, done, info = env.step(action)

        cum_reward += reward
        total_energy_kwh += info["energy_kwh"]
        total_energy_cost += info["energy_cost"]
        total_served += info["served_requests"]
        total_requests += info["total_requests"]
        total_violations += info["service_violations"]
        total_switches += info["mode_switched"]
        latencies.append(info["latency_sec"])
        action_counts[action] += 1

        h = info["hour"]
        hourly_power_kw[h] += (info["power_watts"] / 1000.0) / 4.0
        hourly_users[h] += info["served_requests"] / 4.0
        hourly_actions[h] += action / 4.0

        state_idx = next_state_idx

    service_rate_pct = (
        100.0 * total_served / max(1, total_requests)
    )

    return {
        "cumulative_reward": float(cum_reward),
        "energy_kwh": float(total_energy_kwh),
        "energy_cost": float(total_energy_cost),
        "served_requests": int(total_served),
        "total_requests": int(total_requests),
        "service_rate_pct": float(service_rate_pct),
        "service_violations": int(total_violations),
        "mean_latency_sec": float(np.mean(latencies)),
        "mode_switches": int(total_switches),
        "action_fractions": [c / 96.0 for c in action_counts],
        "hourly_power_kw": hourly_power_kw.tolist(),
        "hourly_users": hourly_users.tolist(),
        "hourly_actions": hourly_actions.tolist(),
    }


def run_evaluation(
    lab_config: LabConfig = DEFAULT_LAB_CONFIG,
    train_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained RL policies and all baseline policies across held-out seeds.
    """
    q_agent = QLearningAgent.load(train_config.model_save_path)
    policies: Dict[str, BasePolicy] = {"Q-Learning": q_agent}

    if Path(train_config.sarsa_model_save_path).exists():
        sarsa_agent = SARSAAgent.load(train_config.sarsa_model_save_path)
        sarsa_agent.name = "SARSA"
        policies["SARSA"] = sarsa_agent

    baselines = get_baseline_policies(seed=train_config.eval_base_seed)
    policies.update(baselines)

    env = ComputerLabEnv(config=lab_config)

    metric_keys = [
        "cumulative_reward",
        "energy_kwh",
        "energy_cost",
        "served_requests",
        "service_rate_pct",
        "service_violations",
        "mean_latency_sec",
        "mode_switches",
    ]

    summary: Dict[str, Any] = {
        "evaluation_protocol": {
            "num_eval_seeds": train_config.num_eval_seeds,
            "episodes_per_seed": train_config.episodes_per_eval_seed,
            "total_episodes_per_policy": (
                train_config.num_eval_seeds * train_config.episodes_per_eval_seed
            ),
            "eval_base_seed": train_config.eval_base_seed,
            "steps_per_episode": lab_config.steps_per_episode,
        },
        "policies": {},
    }

    per_seed_rows: List[Dict[str, Any]] = []

    for policy_name, policy in policies.items():
        seed_means: Dict[str, List[float]] = {k: [] for k in metric_keys}
        all_action_fracs: List[List[float]] = []
        all_hourly_power: List[List[float]] = []
        all_hourly_users: List[List[float]] = []
        all_hourly_actions: List[List[float]] = []

        for s_idx in range(train_config.num_eval_seeds):
            seed_val = train_config.eval_base_seed + s_idx
            ep_metrics: Dict[str, List[float]] = {k: [] for k in metric_keys}

            for ep_offset in range(train_config.episodes_per_eval_seed):
                # Deterministic mix: 4 weekdays + 1 weekend per 5-episode seed block
                day_type = 1 if (ep_offset == 4) else 0
                ep_seed = seed_val * 1000 + ep_offset
                res = evaluate_policy_on_episode(env, policy, ep_seed, day_type)

                for k in metric_keys:
                    ep_metrics[k].append(float(res[k]))
                all_action_fracs.append(res["action_fractions"])
                all_hourly_power.append(res["hourly_power_kw"])
                all_hourly_users.append(res["hourly_users"])
                all_hourly_actions.append(res["hourly_actions"])

            row = {"policy": policy_name, "seed": seed_val}
            for k in metric_keys:
                mean_val = float(np.mean(ep_metrics[k]))
                seed_means[k].append(mean_val)
                row[k] = round(mean_val, 4)
            per_seed_rows.append(row)

        policy_stats: Dict[str, Any] = {}
        for k in metric_keys:
            arr = np.array(seed_means[k], dtype=np.float64)
            policy_stats[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        policy_stats["action_distribution"] = {
            "SLEEP": float(np.mean([a[0] for a in all_action_fracs]) * 100.0),
            "IDLE": float(np.mean([a[1] for a in all_action_fracs]) * 100.0),
            "ACTIVE": float(np.mean([a[2] for a in all_action_fracs]) * 100.0),
        }
        policy_stats["hourly_profile"] = {
            "power_kw": np.mean(all_hourly_power, axis=0).tolist(),
            "served_users": np.mean(all_hourly_users, axis=0).tolist(),
            "mean_action": np.mean(all_hourly_actions, axis=0).tolist(),
        }

        summary["policies"][policy_name] = policy_stats

    # Save JSON summary and per-seed CSV
    Path(train_config.evaluation_results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_config.evaluation_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(train_config.per_seed_results_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["policy", "seed"] + metric_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    if verbose:
        print("=" * 98)
        print(
            f"Evaluation Summary ({train_config.num_eval_seeds} Held-Out Seeds x "
            f"{train_config.episodes_per_eval_seed} Episodes = "
            f"{train_config.num_eval_seeds * train_config.episodes_per_eval_seed} Episodes/Policy)"
        )
        print("=" * 98)
        header = (
            f"{'Policy':<15} | {'Reward':>14} | {'Energy (kWh)':>13} | "
            f"{'Cost ($)':>9} | {'Served':>11} | {'Violations':>10} | "
            f"{'Latency (s)':>11} | {'Switches':>8}"
        )
        print(header)
        print("-" * 98)
        for p_name, p_data in summary["policies"].items():
            r_m = p_data["cumulative_reward"]["mean"]
            r_s = p_data["cumulative_reward"]["std"]
            e_m = p_data["energy_kwh"]["mean"]
            c_m = p_data["energy_cost"]["mean"]
            s_m = p_data["served_requests"]["mean"]
            sr_m = p_data["service_rate_pct"]["mean"]
            v_m = p_data["service_violations"]["mean"]
            l_m = p_data["mean_latency_sec"]["mean"]
            sw_m = p_data["mode_switches"]["mean"]
            print(
                f"{p_name:<15} | {r_m:7.2f}±{r_s:5.2f} | {e_m:13.2f} | "
                f"{c_m:9.2f} | {s_m:5.1f} ({sr_m:4.1f}%) | {v_m:10.2f} | "
                f"{l_m:11.2f} | {sw_m:8.2f}"
            )
        print("=" * 98)
        print(f"Saved JSON summary to: {train_config.evaluation_results_path}")
        print(f"Saved per-seed CSV to: {train_config.per_seed_results_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent and baseline policies."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_TRAINING_CONFIG.num_eval_seeds,
        help="Number of held-out evaluation seeds (default: 20)",
    )
    args = parser.parse_args()

    eval_cfg = TrainingConfig(num_eval_seeds=args.seeds)
    run_evaluation(DEFAULT_LAB_CONFIG, eval_cfg, verbose=True)


if __name__ == "__main__":
    main()
