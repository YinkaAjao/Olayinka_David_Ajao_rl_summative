import gymnasium as gym
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import environment  # Triggers the custom environment registration

# 1. Define the Policy Neural Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        probs = self.fc(x)
        return Categorical(probs)

# 2. Custom Evaluation Function (since we aren't using Stable-Baselines3 here)
def evaluate_policy(policy, env, n_eval_episodes=10):
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_tensor)
                action = dist.sample().item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

# 3. Training Loop and Sweeps
def run_reinforce_sweeps():
    print("Starting PyTorch REINFORCE Hyperparameter Sweeps...")
    
    env = gym.make("ScoutRecommend-v0")
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)

    # 10 distinct hyperparameter combinations
    hyperparams = [
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_dim": 64, "episodes": 600},
        {"learning_rate": 5e-4, "gamma": 0.99, "hidden_dim": 128, "episodes": 600},
        {"learning_rate": 1e-4, "gamma": 0.95, "hidden_dim": 64, "episodes": 800},
        {"learning_rate": 1e-3, "gamma": 0.90, "hidden_dim": 32, "episodes": 600},
        {"learning_rate": 5e-3, "gamma": 0.99, "hidden_dim": 64, "episodes": 500},
        {"learning_rate": 5e-4, "gamma": 0.95, "hidden_dim": 128, "episodes": 600},
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_dim": 128, "episodes": 600},
        {"learning_rate": 1e-4, "gamma": 0.90, "hidden_dim": 64, "episodes": 800},
        {"learning_rate": 5e-3, "gamma": 0.95, "hidden_dim": 32, "episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.90, "hidden_dim": 64, "episodes": 600},
    ]
    
    results = []
    obs_dim = env.observation_space.shape[0]
    act_dim = int(env.action_space.n)

    for i, hp in enumerate(hyperparams):
        print(f"\nREINFORCE Run {i+1}/10 | Params: {hp}")
        
        policy = PolicyNetwork(obs_dim, act_dim, hp["hidden_dim"])
        optimizer = optim.Adam(policy.parameters(), lr=hp["learning_rate"])
        
        # Training phase
        for ep in range(hp["episodes"]):
            obs, _ = env.reset()
            log_probs = []
            rewards = []
            done = False
            
            # Play through one full episode (trajectory)
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = policy(obs_tensor)
                action = dist.sample()
                
                log_prob = dist.log_prob(action)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                log_probs.append(log_prob)
                rewards.append(reward)
            
            # Compute discounted returns (rewards-to-go)
            discounted_returns = []
            cumulative = 0
            for r in reversed(rewards):
                cumulative = r + hp["gamma"] * cumulative
                discounted_returns.insert(0, cumulative)
                
            discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
            
            # Normalize returns to reduce gradient variance (stabilizes training)
            if len(discounted_returns) > 1 and discounted_returns.std() > 0:
                discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
            
            # Calculate Policy Gradient Loss
            loss = []
            for log_p, G in zip(log_probs, discounted_returns):
                loss.append(-log_p * G)
            loss = torch.cat(loss).sum()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Evaluation phase
        mean_reward, std_reward = evaluate_policy(policy, env)
        print(f"Result: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Save PyTorch model state
        torch.save(policy.state_dict(), f"models/pg/reinforce_run_{i+1}.pth")
        
        hp["mean_reward"] = round(mean_reward, 2)
        hp["std_reward"] = round(std_reward, 2)
        results.append(hp)
        
    # Save table for report
    df = pd.DataFrame(results)
    df.to_csv("logs/reinforce_results.csv", index=False)
    print("\nSuccessfully saved all 10 REINFORCE results to logs/reinforce_results.csv")

if __name__ == "__main__":
    run_reinforce_sweeps()