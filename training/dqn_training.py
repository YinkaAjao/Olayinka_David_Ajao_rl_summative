import gymnasium as gym
import pandas as pd
import os
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import environment  # Triggers the custom environment registration

def run_dqn_sweeps():
    print("Starting DQN Hyperparameter Sweeps...")
    
    # Instantiate the environment (no human rendering during training to maximize speed)
    env = gym.make("ScoutRecommend-v0")

    # 10 distinct hyperparameter combinations for DQN
    hyperparams = [
        {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "buffer_size": 10000, "exploration_final_eps": 0.05},
        {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64, "buffer_size": 50000, "exploration_final_eps": 0.10},
        {"learning_rate": 1e-3, "gamma": 0.95, "batch_size": 128, "buffer_size": 100000, "exploration_final_eps": 0.01},
        {"learning_rate": 1e-4, "gamma": 0.90, "batch_size": 32, "buffer_size": 10000, "exploration_final_eps": 0.10},
        {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 256, "buffer_size": 50000, "exploration_final_eps": 0.05},
        {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64, "buffer_size": 10000, "exploration_final_eps": 0.20},
        {"learning_rate": 1e-4, "gamma": 0.95, "batch_size": 128, "buffer_size": 100000, "exploration_final_eps": 0.05},
        {"learning_rate": 5e-4, "gamma": 0.90, "batch_size": 32, "buffer_size": 50000, "exploration_final_eps": 0.01},
        {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 256, "buffer_size": 100000, "exploration_final_eps": 0.10},
        {"learning_rate": 5e-5, "gamma": 0.99, "batch_size": 64, "buffer_size": 10000, "exploration_final_eps": 0.05},
    ]

    results = []
    
    # Ensure output directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/dqn", exist_ok=True)

    for i, hp in enumerate(hyperparams):
        print(f"\nRun {i+1}/10 | Params: {hp}")
        
        # Initialize the DQN model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            batch_size=hp["batch_size"],
            buffer_size=hp["buffer_size"],
            exploration_final_eps=hp["exploration_final_eps"],
            verbose=0
        )

        # Train for 20,000 timesteps
        model.learn(total_timesteps=20000)

        # Evaluate the trained policy over 10 episodes
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Result: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save the model artifact
        model.save(f"models/dqn/dqn_run_{i+1}")

        # Record results for the final report table
        hp["mean_reward"] = round(mean_reward, 2)
        hp["std_reward"] = round(std_reward, 2)
        results.append(hp)

    # Save all hyperparameter results to a CSV for easy report generation
    df = pd.DataFrame(results)
    df.to_csv("logs/dqn_results.csv", index=False)
    print("\nSuccessfully saved all 10 DQN results to logs/dqn_results.csv")

if __name__ == "__main__":
    run_dqn_sweeps()