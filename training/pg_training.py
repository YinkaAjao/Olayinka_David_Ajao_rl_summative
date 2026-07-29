import gymnasium as gym
import pandas as pd
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import environment  # Triggers the custom environment registration

def run_pg_sweeps():
    print("Starting Policy Gradient (PPO & A2C) Hyperparameter Sweeps...")
    
    # Instantiate and wrap the environment with Monitor to silence the SB3 evaluation warning
    env = gym.make("ScoutRecommend-v0")
    env = Monitor(env)

    # Ensure output directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)

    # ---------------------------------------------------------
    # 1. PPO Sweeps
    # ---------------------------------------------------------
    print("\n--- Running PPO Sweeps ---")
    ppo_hyperparams = [
        {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "ent_coef": 0.0, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.01, "gamma": 0.99},
        {"learning_rate": 5e-4, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.0, "gamma": 0.95},
        {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256, "ent_coef": 0.05, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 64, "ent_coef": 0.0, "gamma": 0.90},
        {"learning_rate": 5e-4, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.01, "gamma": 0.99},
        {"learning_rate": 3e-4, "n_steps": 4096, "batch_size": 256, "ent_coef": 0.0, "gamma": 0.95},
        {"learning_rate": 1e-3, "n_steps": 2048, "batch_size": 64, "ent_coef": 0.05, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.0, "gamma": 0.90},
        {"learning_rate": 5e-4, "n_steps": 4096, "batch_size": 64, "ent_coef": 0.01, "gamma": 0.99},
    ]
    
    ppo_results = []
    for i, hp in enumerate(ppo_hyperparams):
        print(f"PPO Run {i+1}/10 | Params: {hp}")
        
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=hp["learning_rate"], 
            n_steps=hp["n_steps"], 
            batch_size=hp["batch_size"], 
            ent_coef=hp["ent_coef"], 
            gamma=hp["gamma"], 
            verbose=0
        )
        
        model.learn(total_timesteps=20000)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        model.save(f"models/pg/ppo_run_{i+1}")
        
        hp["mean_reward"] = round(mean_reward, 2)
        hp["std_reward"] = round(std_reward, 2)
        ppo_results.append(hp)
        
    pd.DataFrame(ppo_results).to_csv("logs/ppo_results.csv", index=False)
    print("Successfully saved 10 PPO results to logs/ppo_results.csv")

    # ---------------------------------------------------------
    # 2. A2C Sweeps
    # ---------------------------------------------------------
    print("\n--- Running A2C Sweeps ---")
    
    # A2C doesn't use batch_size the same way PPO does, it processes the entire rollout (n_steps)
    a2c_hyperparams = [
        {"learning_rate": 7e-4, "n_steps": 5, "ent_coef": 0.0, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 10, "ent_coef": 0.01, "gamma": 0.99},
        {"learning_rate": 5e-4, "n_steps": 5, "ent_coef": 0.0, "gamma": 0.95},
        {"learning_rate": 7e-4, "n_steps": 20, "ent_coef": 0.05, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 5, "ent_coef": 0.0, "gamma": 0.90},
        {"learning_rate": 5e-4, "n_steps": 10, "ent_coef": 0.01, "gamma": 0.99},
        {"learning_rate": 7e-4, "n_steps": 20, "ent_coef": 0.0, "gamma": 0.95},
        {"learning_rate": 1e-3, "n_steps": 5, "ent_coef": 0.05, "gamma": 0.99},
        {"learning_rate": 1e-4, "n_steps": 10, "ent_coef": 0.0, "gamma": 0.90},
        {"learning_rate": 5e-4, "n_steps": 20, "ent_coef": 0.01, "gamma": 0.99},
    ]
    
    a2c_results = []
    for i, hp in enumerate(a2c_hyperparams):
        print(f"A2C Run {i+1}/10 | Params: {hp}")
        
        model = A2C(
            "MlpPolicy", 
            env, 
            learning_rate=hp["learning_rate"], 
            n_steps=hp["n_steps"], 
            ent_coef=hp["ent_coef"], 
            gamma=hp["gamma"], 
            verbose=0
        )
        
        model.learn(total_timesteps=20000)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        model.save(f"models/pg/a2c_run_{i+1}")
        
        hp["mean_reward"] = round(mean_reward, 2)
        hp["std_reward"] = round(std_reward, 2)
        a2c_results.append(hp)
        
    pd.DataFrame(a2c_results).to_csv("logs/a2c_results.csv", index=False)
    print("Successfully saved 10 A2C results to logs/a2c_results.csv")

if __name__ == "__main__":
    run_pg_sweeps()