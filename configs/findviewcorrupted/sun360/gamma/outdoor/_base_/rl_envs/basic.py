_base_ = [
    '../defaults.py',
]
rl_env = dict(
    name="basic",
    success_reward=10.0,
    slack_reward=-0.01,
)
