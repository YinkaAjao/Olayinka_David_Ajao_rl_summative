import time
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
import environment  # Triggers environment registration

def run_best_agent():
    # Read PPO results to locate the best checkpoint dynamically
    try:
        ppo_df = pd.read_csv("logs/ppo_results.csv")
        best_run = ppo_df["mean_reward"].idxmax() + 1
        model_path = f"models/pg/ppo_run_{best_run}"
        print(f"Loading Best Agent Checkpoint: PPO Run {best_run}...")
    except Exception:
        model_path = "models/pg/ppo_run_1"
        print("Defaulting to models/pg/ppo_run_1...")

    # Create the environment with human rendering enabled
    env = gym.make("ScoutRecommend-v0", render_mode="human")
    model = PPO.load(model_path)

    print("\n--- Starting Live Scout Recommendation Demonstration ---")
    
    try:
        # Run for 3 full episodes to stretch the simulation to ~3 minutes
        for episode in range(1, 4):
            obs, info = env.reset()
            env.render()
            total_reward = 0
            
            print(f"\n=========================================")
            print(f"          STARTING EPISODE {episode}           ")
            print(f"=========================================")
            
            for step in range(50):
                # The trained agent determines the best action deterministically
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Pass live updates to Pygame visualizer
                env.unwrapped.visualizer.render_step(info)

                # Verbose terminal output for your video recording
                action_str = "RECOMMEND" if action == 1 else "SKIP"
                print(f"Ep {episode} | Step {step+1:02d} | Action: {action_str:<9} | Match Score: {info['match_score']*100:5.1f}% | Reward: {reward:+4.1f} | Cumulative: {total_reward:+5.1f}")

                # 1.2 seconds x 50 steps = 60 seconds (1 minute per episode)
                time.sleep(1.2)  

                if terminated or truncated:
                    print(f"\nEpisode {episode} Completed! Total Session Reward: {total_reward:.2f}")
                    time.sleep(2)  # Pause for 2 seconds before the next episode starts
                    break

    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_best_agent()