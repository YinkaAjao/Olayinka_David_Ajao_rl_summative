from gymnasium.envs.registration import register

register(
    id="ScoutRecommend-v0",
    entry_point="environment.custom_env:ScoutRecommendationEnv",
)