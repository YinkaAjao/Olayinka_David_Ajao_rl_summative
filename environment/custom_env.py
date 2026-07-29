import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.rendering import ScoutVisualizer

class ScoutRecommendationEnv(gym.Env):
    """
    Custom Environment for ScoutConnect Recommendation Engine.
    The agent decides whether to Recommend (1) or Skip (0) an athlete profile
    based on the athlete's attributes and the scout's historical preferences.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.visualizer = None
        
        # Action Space: 0 = Skip, 1 = Recommend
        self.action_space = spaces.Discrete(2)
        
        # Observation Space: 6 athlete traits + 6 scout preference weights
        # Traits: Speed, Endurance, Technical, Shooting, Defense, Physicality
        # All values normalized between 0.0 and 1.0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Session constraints
        self.current_step = 0
        self.max_steps = 50  # Number of profiles reviewed per session
        
        self.state = None
        self.current_scout_pref = None
        self.current_athlete = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize a synthetic scout preference vector for this episode
        # Ensures weights sum to 1.0
        raw_prefs = self.np_random.uniform(low=0.1, high=1.0, size=(6,))
        self.current_scout_pref = raw_prefs / np.sum(raw_prefs)
        
        self._generate_next_state()
        
        return self.state, {}

    def _generate_next_state(self):
        # Generate random athlete profile (6 physical/technical traits)
        self.current_athlete = self.np_random.uniform(low=0.0, high=1.0, size=(6,))
        
        # Concatenate into the 12D observation array
        self.state = np.concatenate((self.current_athlete, self.current_scout_pref), dtype=np.float32)

    def step(self, action):
        # Core mechanics: Calculate similarity between athlete and scout preferences
        similarity = np.dot(self.current_athlete, self.current_scout_pref)
        
        # Inject Gaussian noise to simulate human unpredictability
        noise = self.np_random.normal(0, 0.05)
        match_score = np.clip(similarity + noise, 0.0, 1.0)

        reward = 0.0
        
        # Reward Oracle
        if action == 1:  # Recommend
            if match_score > 0.85:
                reward = 10.0  # High match -> Scout requests contact
            elif match_score > 0.70:
                reward = 5.0   # Solid match -> Scout adds to shortlist
            elif match_score > 0.50:
                reward = 1.0   # Marginal match -> Scout views profile
            else:
                reward = -2.0  # Poor match -> Recommendation ignored (penalty)
        else:  # Skip
            if match_score < 0.50:
                reward = 0.5   # Correctly skipped a bad fit
            elif match_score > 0.70:
                reward = -5.0  # Missed a high-value candidate (penalty)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        if not terminated:
            self._generate_next_state()

        # Pass metadata for visualization
        info = {
            "match_score": float(match_score),
            "athlete_traits": self.current_athlete.tolist(),
            "scout_prefs": self.current_scout_pref.tolist(),
            "action_taken": "Recommend" if action == 1 else "Skip",
            "reward_earned": reward
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.visualizer is None:
                self.visualizer = ScoutVisualizer()
            
            # Reconstruct the info dict from the current state to pass to the visualizer
            similarity = np.dot(self.current_athlete, self.current_scout_pref)
            info = {
                "match_score": float(similarity), 
                "athlete_traits": self.current_athlete.tolist(),
                "scout_prefs": self.current_scout_pref.tolist(),
                "action_taken": "Viewing",  # Placeholder until action is passed
                "reward_earned": 0.0
            }
            self.visualizer.render_step(info)

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None