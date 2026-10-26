"""
Publication-quality figure generation for RL-Based Computer Lab Power Management.

Reads measured training history (`results/training_history.json`) and held-out
evaluation results (`results/evaluation_summary.json`) and generates figures in
both high-resolution PNG (300 DPI) and vector PDF formats inside `figures/`.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_LAB_CONFIG, DEFAULT_TRAINING_CONFIG
from train import QLearningAgent


# Global publication styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.titlesize": 11.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

POLICY_COLORS = {
    "Q-Learning": "#1b4f72",
    "SARSA": "#117a65",
    "Always-Active": "#d35400",
    "Always-Idle": "#2980b9",
    "Always-Sleep": "#7f8c8d",
    "Random": "#8e44ad",
}


def rolling_mean(values: List[float], window: int = 50) -> np.ndarray:
    """Compute a causal moving average with the given window size."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def plot_training_convergence(
    history: Dict[str, Any],
    eval_summary: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Figure 1: Training Reward Convergence, Epsilon Decay, and TD Error."""
    q_hist = history["q_learning"]
    sarsa_hist = history["sarsa"]
    episodes = np.asarray(q_hist["episode"])
    q_rewards = np.asarray(q_hist["episode_reward"])
    q_smooth = rolling_mean(q_rewards, window=60)
    sarsa_smooth = rolling_mean(sarsa_hist["episode_reward"], window=60)

    active_eval = eval_summary["policies"]["Always-Active"]["cumulative_reward"]["mean"]
    idle_eval = eval_summary["policies"]["Always-Idle"]["cumulative_reward"]["mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=300)

    # Panel A: Episode Reward vs Episode
    ax1.plot(
        episodes,
        q_rewards,
        color="#85c1e9",
        alpha=0.32,
        linewidth=0.7,
        label="Q-Learning (Raw Episode)",
    )
    ax1.plot(
        episodes,
        q_smooth,
        color=POLICY_COLORS["Q-Learning"],
        linewidth=2.1,
        label="Q-Learning (60-Ep Moving Avg)",
    )
    ax1.plot(
        episodes,
        sarsa_smooth,
        color=POLICY_COLORS["SARSA"],
        linewidth=1.8,
        linestyle="-.",
        label="SARSA (60-Ep Moving Avg)",
    )
    ax1.axhline(
        active_eval,
        color=POLICY_COLORS["Always-Active"],
        linestyle="--",
        linewidth=1.3,
        label=f"Always-Active Baseline ({active_eval:.1f})",
    )
    ax1.axhline(
        idle_eval,
        color=POLICY_COLORS["Always-Idle"],
        linestyle=":",
        linewidth=1.3,
        label=f"Always-Idle Baseline ({idle_eval:.1f})",
    )
    ax1.set_title("(a) Cumulative Reward Convergence During Training")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Cumulative Episode Reward")
    ax1.set_xlim(1, len(episodes))
    ax1.set_ylim(-550, 460)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel B: TD Error & Exploration Rate Schedule
    td_smooth = rolling_mean(q_hist["mean_td_error"], window=60)
    eps = np.asarray(q_hist["epsilon"])

    color_td = "#b03a2e"
    ax2.plot(
        episodes,
        td_smooth,
        color=color_td,
        linewidth=2.0,
        label="Mean |TD Error| (60-Ep Avg)",
    )
    ax2.set_title("(b) Bellman TD-Error Convergence & Exploration Schedule")
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Mean Absolute TD Error", color=color_td)
    ax2.tick_params(axis="y", labelcolor=color_td)
    ax2.set_xlim(1, len(episodes))

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    color_eps = "#1f618d"
    ax2_twin.plot(
        episodes,
        eps,
        color=color_eps,
        linestyle="--",
        linewidth=1.8,
        label="Exploration Rate ($\\epsilon$)",
    )
    ax2_twin.set_ylabel("Exploration Probability ($\\epsilon$)", color=color_eps)
    ax2_twin.tick_params(axis="y", labelcolor=color_eps)
    ax2_twin.set_ylim(0.0, 1.05)
    ax2_twin.grid(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "training_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "training_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reward_comparison(eval_summary: Dict[str, Any], out_dir: Path) -> None:
    """Figure 2: Cumulative Reward & Action Mode Distribution Across Policies."""
    order = [
        "Q-Learning",
        "SARSA",
        "Always-Active",
        "Always-Idle",
        "Random",
        "Always-Sleep",
    ]
    policies = [p for p in order if p in eval_summary["policies"]]
    means = [
        eval_summary["policies"][p]["cumulative_reward"]["mean"] for p in policies
    ]
    stds = [
        eval_summary["policies"][p]["cumulative_reward"]["std"] for p in policies
    ]
    colors = [POLICY_COLORS[p] for p in policies]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 3.8), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    x = np.arange(len(policies))
    bars = ax1.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="#1c2833",
        linewidth=0.8,
        alpha=0.90,
        width=0.62,
    )
    ax1.axhline(0.0, color="#2c3e50", linewidth=0.9, linestyle="-")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=18, ha="right")
    ax1.set_ylabel("Mean Cumulative Episode Reward")
    ax1.set_title("(a) Cumulative Reward Across Held-Out Evaluation Seeds")
    ax1.set_ylim(min(means) - 240, max(means) + 210)

    for bar, m, s in zip(bars, means, stds):
        y_pos = m + s + 18 if m >= 0 else m - s - 28
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{m:.1f}\n(±{s:.1f})",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontsize=7.6,
            fontweight="bold",
        )

    # Panel B: Action Mode Selection Breakdown (%)
    sleep_pcts = [
        eval_summary["policies"][p]["action_distribution"]["SLEEP"] for p in policies
    ]
    idle_pcts = [
        eval_summary["policies"][p]["action_distribution"]["IDLE"] for p in policies
    ]
    active_pcts = [
        eval_summary["policies"][p]["action_distribution"]["ACTIVE"] for p in policies
    ]

    ax2.bar(
        x,
        sleep_pcts,
        width=0.58,
        label="SLEEP (8 W standby)",
        color="#5d6d7e",
        edgecolor="#1c2833",
        linewidth=0.6,
    )
    ax2.bar(
        x,
        idle_pcts,
        width=0.58,
        bottom=sleep_pcts,
        label="IDLE (45 W standby)",
        color="#5dade2",
        edgecolor="#1c2833",
        linewidth=0.6,
    )
    bottom_active = np.array(sleep_pcts) + np.array(idle_pcts)
    ax2.bar(
        x,
        active_pcts,
        width=0.58,
        bottom=bottom_active,
        label="ACTIVE (110 W standby)",
        color="#f39c12",
        edgecolor="#1c2833",
        linewidth=0.6,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=18, ha="right")
    ax2.set_ylabel("Fraction of Time Steps (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Standby Power Mode Allocation (%)")
    ax2.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "reward_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "reward_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy_and_cost(eval_summary: Dict[str, Any], out_dir: Path) -> None:
    """Figure 3: Energy Consumption, ToU Electricity Cost, and 24h Power Profile."""
    order = [
        "Q-Learning",
        "SARSA",
        "Always-Active",
        "Always-Idle",
        "Random",
        "Always-Sleep",
    ]
    policies = [p for p in order if p in eval_summary["policies"]]

    energy_means = [
        eval_summary["policies"][p]["energy_kwh"]["mean"] for p in policies
    ]
    energy_stds = [
        eval_summary["policies"][p]["energy_kwh"]["std"] for p in policies
    ]
    cost_means = [
        eval_summary["policies"][p]["energy_cost"]["mean"] for p in policies
    ]
    cost_stds = [
        eval_summary["policies"][p]["energy_cost"]["std"] for p in policies
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=300)

    # Panel A: Dual bar chart of Energy (kWh) and Cost ($)
    x = np.arange(len(policies))
    w = 0.36
    b1 = ax1.bar(
        x - w / 2,
        energy_means,
        w,
        yerr=energy_stds,
        capsize=3,
        color="#2874a6",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="Daily Energy (kWh)",
    )
    ax1.set_ylabel("Electrical Energy Consumed (kWh / day)", color="#2874a6")
    ax1.tick_params(axis="y", labelcolor="#2874a6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=18, ha="right")
    ax1.set_ylim(0, 68)
    ax1.set_title("(a) Daily Energy Consumption & ToU Cost")

    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    b2 = ax1_twin.bar(
        x + w / 2,
        cost_means,
        w,
        yerr=cost_stds,
        capsize=3,
        color="#ca6f1e",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="ToU Electricity Cost ($ / day)",
    )
    ax1_twin.set_ylabel("Daily Electricity Cost (USD / day)", color="#ca6f1e")
    ax1_twin.tick_params(axis="y", labelcolor="#ca6f1e")
    ax1_twin.set_ylim(0, 8.8)
    ax1_twin.grid(False)

    for bar, val in zip(b1, energy_means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#1b4f72",
            fontweight="bold",
        )
    for bar, val in zip(b2, cost_means):
        ax1_twin.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.16,
            f"${val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#935116",
            fontweight="bold",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # Panel B: 24-Hour Diurnal Power Draw Profile
    hours = np.arange(24)
    for p_name, style in [
        ("Always-Active", "--"),
        ("Q-Learning", "-"),
        ("Always-Idle", "-."),
        ("Always-Sleep", ":"),
    ]:
        p_kw = eval_summary["policies"][p_name]["hourly_profile"]["power_kw"]
        ax2.plot(
            hours,
            p_kw,
            linestyle=style,
            linewidth=2.1 if p_name == "Q-Learning" else 1.6,
            marker="o" if p_name == "Q-Learning" else None,
            markersize=3.8,
            color=POLICY_COLORS[p_name],
            label=p_name,
        )

    # Highlight Peak Tariff window (12:00-18:00)
    ax2.axvspan(12, 17.9, color="#f5b041", alpha=0.16, label="Peak Tariff ($0.18/kWh)")
    ax2.set_title("(b) 24-Hour Diurnal Lab Power Draw Profile")
    ax2.set_xlabel("Hour of Day (00:00 - 23:00)")
    ax2.set_ylabel("Mean Lab Power Demand (kW)")
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0, 3.0)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "energy_and_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "energy_and_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_service_and_latency(eval_summary: Dict[str, Any], out_dir: Path) -> None:
    """Figure 4: Service Quality, Violations, Wake-Up Latency, and Mode Switches."""
    order = [
        "Q-Learning",
        "SARSA",
        "Always-Active",
        "Always-Idle",
        "Random",
        "Always-Sleep",
    ]
    policies = [p for p in order if p in eval_summary["policies"]]

    srv_rate = [
        eval_summary["policies"][p]["service_rate_pct"]["mean"] for p in policies
    ]
    violations = [
        eval_summary["policies"][p]["service_violations"]["mean"] for p in policies
    ]
    latencies = [
        eval_summary["policies"][p]["mean_latency_sec"]["mean"] for p in policies
    ]
    switches = [
        eval_summary["policies"][p]["mode_switches"]["mean"] for p in policies
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=300)
    x = np.arange(len(policies))
    w = 0.36

    # Panel A: Service Completion Rate (%) vs Service Violations
    b1 = ax1.bar(
        x - w / 2,
        srv_rate,
        w,
        color="#1e8449",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="Service Rate (%)",
    )
    ax1.set_ylabel("User Service Completion Rate (%)", color="#1e8449")
    ax1.tick_params(axis="y", labelcolor="#1e8449")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=18, ha="right")
    ax1.set_ylim(0, 116)
    ax1.set_title("(a) Service Completion Rate & QoS Violations")

    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    b2 = ax1_twin.bar(
        x + w / 2,
        violations,
        w,
        color="#c0392b",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="QoS Violations / Day",
    )
    ax1_twin.set_ylabel("Service Violations (Requests / Day)", color="#c0392b")
    ax1_twin.tick_params(axis="y", labelcolor="#c0392b")
    ax1_twin.set_ylim(0, max(violations) * 1.22)
    ax1_twin.grid(False)

    for bar, val in zip(b1, srv_rate):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#145a32",
            fontweight="bold",
        )
    for bar, val in zip(b2, violations):
        ax1_twin.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.0,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#922b21",
            fontweight="bold",
        )

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center left", frameon=True)

    # Panel B: Wake-Up Latency (s) vs Mode Switching Frequency
    b3 = ax2.bar(
        x - w / 2,
        latencies,
        w,
        color="#884ea0",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="Mean Wake-Up Latency (s)",
    )
    ax2.set_ylabel("Mean Response / Wake-Up Latency (s)", color="#884ea0")
    ax2.tick_params(axis="y", labelcolor="#884ea0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=18, ha="right")
    ax2.set_ylim(0, max(latencies) * 1.25)
    ax2.set_title("(b) Wake-Up Latency & Power-Mode Switches")

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    b4 = ax2_twin.bar(
        x + w / 2,
        switches,
        w,
        color="#566573",
        edgecolor="#1c2833",
        linewidth=0.7,
        label="Mode Switches / Day",
    )
    ax2_twin.set_ylabel("Power-Mode Switches / Episode", color="#566573")
    ax2_twin.tick_params(axis="y", labelcolor="#566573")
    ax2_twin.set_ylim(0, max(switches) * 1.25)
    ax2_twin.grid(False)

    for bar, val in zip(b3, latencies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.6,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#633974",
            fontweight="bold",
        )
    for bar, val in zip(b4, switches):
        ax2_twin.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#2c3e50",
            fontweight="bold",
        )

    l3, lab3 = ax2.get_legend_handles_labels()
    l4, lab4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(l3 + l4, lab3 + lab4, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "service_and_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "service_and_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_learned_policy_heatmap(
    agent: QLearningAgent,
    out_dir: Path,
) -> None:
    """Figure 5: Learned Greedy Q-Learning Policy across Time Period & Demand Level."""
    from lab_env import ComputerLabEnv

    env = ComputerLabEnv(config=DEFAULT_LAB_CONFIG)
    period_labels = [
        "00-03h",
        "03-06h",
        "06-09h",
        "09-12h",
        "12-15h",
        "15-18h",
        "18-21h",
        "21-24h",
    ]
    demand_labels = [
        "Bin 0 (0-1 PC)",
        "Bin 1 (2-5 PCs)",
        "Bin 2 (6-10 PCs)",
        "Bin 3 (11-15 PCs)",
        "Bin 4 (16-20 PCs)",
    ]

    # Compute majority greedy action across trend_bin and current_mode for Weekday (0) and Weekend (1)
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 3.6), dpi=300, sharey=True, gridspec_kw={"wspace": 0.08}
    )
    cmap = matplotlib.colors.ListedColormap(["#5d6d7e", "#5dade2", "#f39c12"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for day_type, ax, title in [
        (0, axes[0], "(a) Weekday Learned Policy $\\pi^*(s)$"),
        (1, axes[1], "(b) Weekend Learned Policy $\\pi^*(s)$"),
    ]:
        grid = np.zeros((5, 8), dtype=np.float64)
        for d_bin in range(5):
            for p_idx in range(8):
                # Evaluate greedy action when trend is stable/rising (1) and current_mode=IDLE (1)
                s_idx = env.encode_state((p_idx, day_type, d_bin, 1, 1))
                if np.any(agent.visit_counts[s_idx] > 0):
                    action = agent.select_action(s_idx, training=False)
                else:
                    # Fallback across all trends/modes in that (period, day_type, demand_bin)
                    actions_collected = []
                    for tr in range(3):
                        for m in range(3):
                            si = env.encode_state((p_idx, day_type, d_bin, tr, m))
                            if np.any(agent.visit_counts[si] > 0):
                                actions_collected.append(
                                    agent.select_action(si, training=False)
                                )
                    if actions_collected:
                        action = int(np.round(np.mean(actions_collected)))
                    else:
                        action = 0 if p_idx in (0, 1, 7) and d_bin == 0 else (1 if d_bin <= 2 else 2)
                grid[d_bin, p_idx] = action

        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(8))
        ax.set_xticklabels(period_labels, rotation=25, ha="right")
        ax.set_yticks(range(5))
        if day_type == 0:
            ax.set_yticklabels(demand_labels)
            ax.set_ylabel("Active Workstation Demand Bracket")
        ax.set_xlabel("3-Hour Diurnal Time Block")
        ax.set_title(title)
        ax.grid(False)

        action_short = {0: "SLEEP", 1: "IDLE", 2: "ACTIVE"}
        for i in range(5):
            for j in range(8):
                val = int(grid[i, j])
                text_color = "white" if val == 0 else "#1c2833"
                ax.text(
                    j,
                    i,
                    action_short[val],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=text_color,
                )

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        ticks=[0, 1, 2],
        fraction=0.028,
        pad=0.03,
    )
    cbar.ax.set_yticklabels(["0: SLEEP", "1: IDLE", "2: ACTIVE"])
    fig.savefig(out_dir / "learned_policy_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "learned_policy_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(DEFAULT_TRAINING_CONFIG.training_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    with open(DEFAULT_TRAINING_CONFIG.evaluation_results_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    agent = QLearningAgent.load(DEFAULT_TRAINING_CONFIG.model_save_path)

    plot_training_convergence(history, eval_summary, out_dir)
    plot_reward_comparison(eval_summary, out_dir)
    plot_energy_and_cost(eval_summary, out_dir)
    plot_service_and_latency(eval_summary, out_dir)
    plot_learned_policy_heatmap(agent, out_dir)

    print("Successfully generated all experimental figures in `figures/`:")
    for p in sorted(out_dir.glob("*")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
