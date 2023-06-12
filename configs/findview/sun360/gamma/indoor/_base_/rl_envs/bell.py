_base_ = [
    "../defaults.py",
]
rl_env = dict(
    name="bell",
    success_reward=100.0,
    slack_reward=-0.01,
    param=10,
)
