_base_ = [
    "../defaults.py",
]
rl_env = dict(
    name="inverse",
    success_reward=100.0,
    slack_reward=-0.01,
    param=0.1,
)
