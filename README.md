# RL Scout Recommendation Engine

**Author:** Olayinka David Ajao  
**Project:** Reinforcement Learning Summative Assignment

## Project Overview
This repository contains a custom Reinforcement Learning environment designed to solve an information overload problem in sports analytics: prioritizing athlete profiles for talent scouts. 

Instead of a generic grid-world, this environment acts as a sequential triage system. At each timestep, the agent observes a 12-dimensional continuous state representing an athlete's traits and a scout's historical preference weights. The agent must then make a discrete binary decision to either **Recommend** or **Skip** the profile. The reward oracle uses cosine similarity injected with Gaussian noise to simulate human unpredictability, rewarding the agent for high-value matches and penalizing it for missing top talent or spamming the scout with noise.

## Setup & Installation (Strictly via `uv`)

This project uses `uv` for seamless, deterministic dependency management. **No manual virtual environment creation is required.**

1. **Clone the repository:**
   ```bash
   git clone [<your-github-repo-link>](https://github.com/YinkaAjao/Olayinka_David_Ajao_rl_summative.git)
   cd olayinka_david_ajao_rl_summative

   ```

2. **Sync the environment:**

This command reads the `pyproject.toml`, instantly builds the isolated environment, and installs all necessary packages.
```bash
uv sync

```



## Running the Visual Demonstration

To view the fully trained agent acting within the custom Pygame GUI, run the main execution file. This bypasses the need for manual activation:

```bash
uv run main.py

```

*Note: This will launch a Pygame window rendering a dynamic Trait Alignment Radar, animating the agent's real-time triage decisions while outputting verbose logs to your terminal.*

## Training & Hyperparameter Sweeps

The project includes four distinct Reinforcement Learning algorithms. Each algorithm was trained across 10 distinct hyperparameter combinations (40 sweeps total). You can execute the training sweeps individually using the following commands:

**1. Deep Q-Network (DQN) Sweep:**

```bash
uv run python -m training.dqn_training

```

**2. PPO & A2C Sweeps:**

```bash
uv run python -m training.pg_training

```

**3. REINFORCE Sweep (Custom PyTorch Implementation):**

```bash
uv run python -m training.reinforce_training

```

## Generating Results & Plots

Once training is complete, the logs are exported as CSV files into the `logs/` directory. To process these logs, evaluate generalization on unseen profiles, and generate the final report artifacts, run:

```bash
uv run python -m training.generate_plots

```

This will populate the `assets/` folder with the cumulative reward comparisons, best-model performance charts, and the generalization tests.

## Repository Structure

```text
olayinka_david_ajao_rl_summative/
├── pyproject.toml              # uv dependency configurations
├── uv.lock                     # Strict lockfile
├── README.md                   # Project documentation
├── main.py                     # Entry point for Live Agent GUI Simulation
├── play.py                     # Alternate showcase script
├── environment/                
│   ├── __init__.py             # Gym environment registration
│   ├── custom_env.py           # Core RL logic and reward oracle
│   └── rendering.py            # Pygame GUI radar chart and dashboard
├── training/
│   ├── __init__.py
│   ├── dqn_training.py         # SB3 DQN 10-run sweep
│   ├── pg_training.py          # SB3 PPO & A2C 10-run sweeps
│   ├── reinforce_training.py   # Custom PyTorch REINFORCE 10-run sweep
│   └── generate_plots.py       # Generalization testing and plot creation
├── logs/                       # Auto-generated CSV metric files
├── models/                     # Auto-generated .pth and SB3 model checkpoints
└── assets/                     # Auto-generated report figures (png)

```

## Results Summary

Across 40 experimental runs, **Proximal Policy Optimization (PPO)** proved to be the most robust and highest-performing algorithm for this environment, managing the simulated noise better than DQN and capturing the highest peak reward of **47.30**. The best PPO checkpoint also outperformed DQN in zero-shot generalization testing across 20 unseen scout profiles.
