_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py'
]
rl_env = dict(
    name="bell",
    success_reward=100.0,
    slack_reward=-0.01,
    param=10,
)
