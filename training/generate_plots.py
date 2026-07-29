import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO, A2C
import environment  # Register environment

def generate_all_artifacts():
    print("Generating report plots and artifacts...")
    os.makedirs("assets", exist_ok=True)
    
    # 1. Load CSV Log Data
    dqn_df = pd.read_csv("logs/dqn_results.csv")
    ppo_df = pd.read_csv("logs/ppo_results.csv")
    a2c_df = pd.read_csv("logs/a2c_results.csv")
    reinforce_df = pd.read_csv("logs/reinforce_results.csv")

    # ---------------------------------------------------------
    # Plot 1: Cumulative Mean Rewards Comparison across Sweeps
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Hyperparameter Tuning Performance (Mean Reward Across 10 Runs)", fontsize=14, fontweight='bold')

    axs[0, 0].plot(dqn_df.index + 1, dqn_df["mean_reward"], marker='o', color='blue', label='DQN')
    axs[0, 0].set_title("DQN Sweeps")
    axs[0, 0].set_ylabel("Mean Reward")
    axs[0, 0].grid(True)

    axs[0, 1].plot(ppo_df.index + 1, ppo_df["mean_reward"], marker='s', color='green', label='PPO')
    axs[0, 1].set_title("PPO Sweeps")
    axs[0, 1].grid(True)

    axs[1, 0].plot(a2c_df.index + 1, a2c_df["mean_reward"], marker='^', color='orange', label='A2C')
    axs[1, 0].set_title("A2C Sweeps")
    axs[1, 0].set_xlabel("Run Number")
    axs[1, 0].set_ylabel("Mean Reward")
    axs[1, 0].grid(True)

    axs[1, 1].plot(reinforce_df.index + 1, reinforce_df["mean_reward"], marker='d', color='purple', label='REINFORCE')
    axs[1, 1].set_title("REINFORCE Sweeps")
    axs[1, 1].set_xlabel("Run Number")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("assets/cumulative_rewards_comparison.png")
    plt.close()
    print("Saved: assets/cumulative_rewards_comparison.png")

    # ---------------------------------------------------------
    # Plot 2: Best Model Comparison Bar Chart
    # ---------------------------------------------------------
    best_dqn = dqn_df["mean_reward"].max()
    best_ppo = ppo_df["mean_reward"].max()
    best_a2c = a2c_df["mean_reward"].max()
    best_reinforce = reinforce_df["mean_reward"].max()

    plt.figure(figsize=(8, 6))
    methods = ["DQN", "PPO", "A2C", "REINFORCE"]
    rewards = [best_dqn, best_ppo, best_a2c, best_reinforce]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    bars = plt.bar(methods, rewards, color=colors)
    plt.ylabel("Best Mean Reward")
    plt.title("Best Model Performance Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.savefig("assets/best_model_comparison.png")
    plt.close()
    print("Saved: assets/best_model_comparison.png")

    # ---------------------------------------------------------
    # Plot 3: Generalization Evaluation on Unseen Environments
    # ---------------------------------------------------------
    print("Evaluating generalization on 20 unseen scout profiles...")
    env = gym.make("ScoutRecommend-v0")
    
    # Identify top run index for PPO and DQN
    best_ppo_idx = ppo_df["mean_reward"].idxmax() + 1
    best_dqn_idx = dqn_df["mean_reward"].idxmax() + 1

    best_ppo_model = PPO.load(f"models/pg/ppo_run_{best_ppo_idx}")
    best_dqn_model = DQN.load(f"models/dqn/dqn_run_{best_dqn_idx}")

    ppo_gen_rewards, dqn_gen_rewards = [], []

    for _ in range(20):
        # PPO Generalization
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = best_ppo_model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        ppo_gen_rewards.append(ep_reward)

        # DQN Generalization
        obs, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            action, _ = best_dqn_model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        dqn_gen_rewards.append(ep_reward)

    plt.figure(figsize=(9, 5))
    plt.plot(ppo_gen_rewards, label="PPO (Best)", color="green", linestyle="-", marker="s")
    plt.plot(dqn_gen_rewards, label="DQN (Best)", color="blue", linestyle="--", marker="o")
    plt.axhline(y=np.mean(ppo_gen_rewards), color="green", linestyle=":", label=f"PPO Avg: {np.mean(ppo_gen_rewards):.1f}")
    plt.axhline(y=np.mean(dqn_gen_rewards), color="blue", linestyle=":", label=f"DQN Avg: {np.mean(dqn_gen_rewards):.1f}")
    plt.title("Generalization Test across 20 Unseen Test Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/generalization_test.png")
    plt.close()
    print("Saved: assets/generalization_test.png")

if __name__ == "__main__":
    generate_all_artifacts()