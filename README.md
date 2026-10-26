<<<<<<< HEAD
# RL-Based University Computer Lab Power Management

**Course:** Artificial Intelligence — Assignment 2: Real-Life Use Cases of Reinforcement Learning (Mini-Project)  
**Institution:** Indian Institute of Technology Jodhpur (IIT Jodhpur)  
**Team Members:**
- **Kunal Singh** (Roll No: `B24CM1041`)
- **Katravath Kurumurthy** (Roll No: `B24CM1035`)

---

## 1. Project Objective & Real-World Motivation

University computer laboratories typically keep tens or hundreds of workstations powered on 24 hours a day to ensure immediate availability for students. However, actual workstation occupancy follows highly non-stationary diurnal and weekly patterns—peaking during scheduled morning/afternoon practical labs and dropping to near zero overnight and on weekend nights. Leaving unoccupied machines in a high-power `ACTIVE` state wastes substantial electrical energy and incurs high Time-of-Use (ToU) electricity costs, whereas aggressively forcing machines into deep `SLEEP` (Suspend-to-RAM S3) degrades user experience through long wake-up latencies and login timeouts during arrival surges.

This project formulates university computer lab power management as a sequential decision-making **Markov Decision Process (MDP)** and trains **Tabular Q-Learning** (off-policy TD control) and **Tabular SARSA** (on-policy TD control) agents to dynamically select optimal standby power modes (`SLEEP`, `IDLE`, or `ACTIVE`) every 15 minutes across a 20-workstation lab.

---

## 2. MDP Formulation

### 2.1 State Space ($\mathcal{S}$)
At each 15-minute decision step $t \in \{0, 1, \dots, 95\}$ ($T = 96$ steps per 24-hour episode), the agent observes a 5-dimensional discrete state tuple:
$$s_t = (p_t, \, d_t, \, u_t, \, \delta_t, \, m_t) \in \mathcal{S}, \quad \dim(\mathcal{S}) = 5$$
1. **Diurnal Time Block ($p_t \in \{0, 1, \dots, 7\}$):** 8 three-hour blocks across the 24-hour cycle ($p_t = \lfloor h_t / 3 \rfloor$: `00-03h`, `03-06h`, `06-09h`, `09-12h`, `12-15h`, `15-18h`, `18-21h`, `21-24h`).
2. **Day Type ($d_t \in \{0, 1\}$):** `0 = Weekday` (scheduled practical labs + self-study), `1 = Weekend` (open project usage).
3. **Occupancy Demand Level ($u_t \in \{0, 1, 2, 3, 4\}$):** Discretized active workstation count out of $N = 20$ computers:
   - Bin 0 (Very Low): $0\text{--}1$ active PCs
   - Bin 1 (Low): $2\text{--}5$ active PCs
   - Bin 2 (Moderate): $6\text{--}10$ active PCs
   - Bin 3 (High): $11\text{--}15$ active PCs
   - Bin 4 (Peak): $16\text{--}20$ active PCs
4. **Demand Trend ($\delta_t \in \{0, 1, 2\}$):** Step-over-step occupancy change $\Delta n_t = n_t - n_{t-1}$:
   - `0 = Falling` ($\Delta n_t \le -2$)
   - `1 = Stable` ($-1 \le \Delta n_t \le +1$)
   - `2 = Rising` ($\Delta n_t \ge +2$)
5. **Current Standby Power Mode ($m_t \in \{0, 1, 2\}$):** Standby pool power state from the previous interval (`0 = SLEEP`, `1 = IDLE`, `2 = ACTIVE`), required to preserve the Markov property under mode-switching energy overheads and wake-up transition delays.

**State Space Cardinality:**
$$|\mathcal{S}| = 8 \times 2 \times 5 \times 3 \times 3 = 720 \text{ discrete states.}$$

### 2.2 Action Space ($\mathcal{A}$)
At each step $t$, the agent selects a discrete power management action $a_t \in \mathcal{A} = \{0, 1, 2\}$:
- **$a_t = 0$ (`SLEEP` — Deep Suspend-to-RAM S3):** Unoccupied workstations draw $P_{\text{sleep}} = 8\text{ W}$. Minimizes standby power but incurs a $45.0\text{ s}$ wake-up latency upon student arrival, causes $50\%$ arrival timeout violations during active requests, and risks premature suspension of paused sessions.
- **$a_t = 1$ (`IDLE` — Low-Power Ready C-State, Display Off):** Unoccupied workstations draw $P_{\text{idle}} = 45\text{ W}$. Smoothly serves moderate arrivals ($\le 3$ arrivals per 15-min slot) at $5.5\text{ s}$ wake-up latency, but experiences queue congestion and partial timeouts during large scheduled-lab arrival surges ($>3$ arrivals/step).
- **$a_t = 2$ (`ACTIVE` — High-Performance P0 Ready State):** Unoccupied workstations draw $P_{\text{active}} = 110\text{ W}$ (while busy PCs draw $P_{\text{busy}} = 150\text{ W}$). Guarantees immediate login ($1.5\text{ s}$ latency) and zero service violations even during peak lab surges, at the expense of high standby power draw.

### 2.3 Reward Function ($\mathcal{R}$)
The immediate scalar reward $R_t$ at step $t$ balances student Quality of Service (QoS) against electrical energy consumption, response latency, and hardware mode-switching overhead:
$$R_t = w_{\text{srv}} \, N_{\text{served}, t} - w_{\text{eng}} \, E_t - w_{\text{viol}} \, N_{\text{viol}, t} - w_{\text{lat}} \left(\frac{L_t}{L_{\text{ref}}}\right) - w_{\text{sw}} \, \mathbb{I}(a_t \neq m_t)$$
where:
- $N_{\text{served}, t}$: Number of active student sessions successfully served during step $t$ ($w_{\text{srv}} = 1.25$).
- $E_t$: Total electrical energy consumed (in $\text{kWh}$) during the 15-minute interval, including active workstations, standby workstations, and mode-transition energy spikes ($w_{\text{eng}} = 6.00\text{ per kWh}$).
- $N_{\text{viol}, t}$: Total QoS service violations (unserved/timed-out arrivals plus premature session suspends, $w_{\text{viol}} = 6.50$).
- $L_t / L_{\text{ref}}$: Normalized user response/wake-up latency in seconds with $L_{\text{ref}} = 10.0\text{ s}$ ($w_{\text{lat}} = 1.50$).
- $\mathbb{I}(a_t \neq m_t)$: Mode-switching indicator penalizing frequent power-state oscillation ($w_{\text{sw}} = 0.70$).

---

## 3. RL Algorithms & Hyperparameters

Given the finite, compact discrete state-action space ($|\mathcal{S}| \times |\mathcal{A}| = 720 \times 3 = 2,160$ table entries), **Tabular Q-Learning** provides exact value representation without function approximation instability, guaranteed convergence under sufficient exploration, and direct policy interpretability. We also implement **Tabular SARSA** as an on-policy TD(0) benchmark.

- **Q-Learning Update (Off-Policy TD Control):**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_k \Big[ R_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]$$
- **SARSA Update (On-Policy TD Control):**
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_k \Big[ R_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \Big]$$

| Hyperparameter | Symbol | Value | Rationale |
| :--- | :---: | :---: | :--- |
| Training Episodes | $K$ | $3,000$ | Provides $288,000$ step transitions across all reachable diurnal states |
| Episode Horizon | $T$ | $96\text{ steps}$ | 24 hours at $\Delta t = 15\text{ min}$ ($0.25\text{ h}$) resolution |
| Discount Factor | $\gamma$ | $0.96$ | Captures multi-hour lookahead for upcoming lab sessions and ToU tariffs |
| Initial Learning Rate | $\alpha_0$ | $0.15$ | Fast initial value propagation across tabular states |
| Minimum Learning Rate | $\alpha_{\min}$ | $0.015$ | Stabilizes Q-value estimates in late training ($\lambda_\alpha = 0.9992$/ep) |
| Initial Exploration Rate | $\epsilon_0$ | $1.00$ | Pure exploration initially, prioritizing unvisited $(s, a)$ pairs |
| Minimum Exploration Rate | $\epsilon_{\min}$ | $0.02$ | Retains $2\%$ exploration floor ($\lambda_\epsilon = 0.9982$/ep) |
| Training Seed | `train_seed` | `42` | Deterministic training trajectory generation |
| Held-Out Eval Seeds | `eval_seeds` | `1000..1019` | 20 independent seeds $\times$ 5 episodes = 100 evaluation episodes/policy |

---

## 4. Repository Structure

```text
.
├── config.py                  # Physical parameters, MDP dimensions, reward weights, hyperparameters
├── lab_env.py                 # 20-PC university computer lab MDP simulation environment
├── baselines.py               # Baseline policies: Always-Active, Always-Idle, Always-Sleep, Random
├── train.py                   # Tabular Q-Learning & SARSA agents, training loop, model serialization
├── evaluate.py                # Multi-seed held-out evaluation pipeline (20 seeds x 5 episodes)
├── plot_results.py            # Publication-quality graph generator (PNG 300 DPI + vector PDF)
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation and reproduction instructions
├── models/
│   ├── q_learning_agent.npz   # Trained Tabular Q-Learning checkpoint
│   └── sarsa_agent.npz        # Trained Tabular SARSA checkpoint
├── results/
│   ├── training_history.json  # Per-episode training metrics across 3,000 episodes
│   ├── evaluation_summary.json# Aggregate held-out evaluation metrics across all policies
│   └── evaluation_per_seed.csv# Per-seed evaluation breakdown across 20 seeds
├── figures/
│   ├── training_convergence.png / .pdf
│   ├── reward_comparison.png / .pdf
│   ├── energy_and_cost.png / .pdf
│   ├── service_and_latency.png / .pdf
│   └── learned_policy_heatmap.png / .pdf
├── report/
│   ├── report.tex             # Two-column LaTeX source for the 2-page academic report
│   └── report.pdf             # Compiled 2-page (+ references) two-column PDF report
└── tests/
    └── test_project.py        # Unit and integration tests (pytest)
```

---

## 5. Installation & Reproduction Instructions

### 5.1 Environment Setup
```bash
python3 -m pip install -r requirements.txt
```

### 5.2 Train the RL Agents
```bash
python3 train.py --episodes 3000 --seed 42
```
**Expected Output:** Trains both `Q-Learning` and `SARSA` for 3,000 episodes ($288,000$ transitions), saves checkpoints to `models/q_learning_agent.npz` and `models/sarsa_agent.npz`, and writes episode histories to `results/training_history.json`.

### 5.3 Evaluate Trained Agents Against Baselines
```bash
python3 evaluate.py --seeds 20
```
**Expected Output:** Evaluates `Q-Learning`, `SARSA`, `Always-Active`, `Always-Idle`, `Always-Sleep`, and `Random` across 20 held-out seeds (100 episodes per policy) and saves `results/evaluation_summary.json` and `results/evaluation_per_seed.csv`.

### 5.4 Generate Experimental Graphs
```bash
python3 plot_results.py
```
**Expected Output:** Generates all 5 figures in both `.png` and `.pdf` formats inside `figures/`.

### 5.5 Run Unit & Integration Tests
```bash
pytest -v
```

---

## 6. Experimental Results (20 Held-Out Seeds $\times$ 5 Episodes = 100 Episodes/Policy)

| Policy | Cumulative Reward ($\uparrow$) | Daily Energy (kWh $\downarrow$) | Daily ToU Cost (USD $\downarrow$) | Served Requests / Day ($\uparrow$) | Service Rate ($\uparrow$) | QoS Violations / Day ($\downarrow$) | Mean Latency (s $\downarrow$) | Mode Switches / Day |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q-Learning (Ours)** | **373.52 $\pm$ 26.64** | **40.31 $\pm$ 0.43** | **\$5.61 $\pm$ 0.06** | **525.4** | **99.6%** | **1.61** | **1.78 s** | **7.50** |
| **SARSA (Ours)** | 367.60 $\pm$ 27.70 | 39.86 $\pm$ 0.45 | \$5.51 $\pm$ 0.06 | 524.7 | 99.5% | 2.08 | 1.97 s | 10.40 |
| **Always-Active** | 298.72 $\pm$ 26.82 | 58.17 $\pm$ 0.37 | \$7.13 $\pm$ 0.05 | 529.8 | 100.0% | 0.00 | 0.96 s | 1.00 |
| **Always-Idle** | 143.74 $\pm$ 22.58 | 33.60 $\pm$ 0.56 | \$4.37 $\pm$ 0.08 | 457.3 | 95.5% | 22.70 | 5.46 s | 0.00 |
| **Random** | -372.49 $\pm$ 40.00 | 37.90 $\pm$ 1.13 | \$4.75 $\pm$ 0.14 | 347.4 | 86.2% | 54.89 | 12.34 s | 64.07 |
| **Always-Sleep** | -1029.45 $\pm$ 24.54 | 9.21 $\pm$ 0.39 | \$1.27 $\pm$ 0.05 | 149.3 | 55.7% | 114.33 | 28.96 s | 1.00 |

### Key Findings & Limitations
1. **Substantial Energy & Cost Savings with Near-Zero QoS Degradation:** `Q-Learning` achieves the highest cumulative reward (**$373.52$**, a **$+25.0\%$** improvement over `Always-Active` and **$+159.9\%$** over `Always-Idle`), reducing daily electricity consumption by **$30.7\%$** ($40.31\text{ kWh}$ vs. $58.17\text{ kWh}$, saving $17.86\text{ kWh/day}$ or $\sim 6,519\text{ kWh/year}$ per 20-PC lab) and reducing ToU electricity cost by **$21.3\%$** ($\$5.61$ vs. $\$7.13$/day) while maintaining a **$99.6\%$** student service completion rate and **$1.78\text{ s}$** average latency.
2. **Interpretable Multi-Regime Policy:** The learned Q-table places the standby fleet into `SLEEP` ($25.2\%$ of steps) during empty overnight blocks (`00:00–06:00`), transitions to `IDLE` ($20.8\%$ of steps) during moderate shoulder/evening hours and weekend study periods, and proactively pre-warms machines into `ACTIVE` ($54.0\%$ of steps) ahead of scheduled weekday lab surges.
3. **Limitations:** The current formulation assumes homogeneous lab workstations and applies a unified power mode to the unoccupied standby pool. Extending the action space to heterogeneous per-row or per-machine cluster control under Deep Q-Networks (DQN) with real-time exam/deadline calendar features is a natural direction for future work.
=======
# rl-lab-power-management
>>>>>>> 9e6dd8aa926c8824933d43f5d20b5d03f48d9d87
