_base_ = [
    '../../../_base_/datasets/sun360_alpha_outdoor.py',
    '../../../_base_/envs/findview.py'
]
rl_env = dict(
    name="basic",
    success_reward=10.0,
    slack_reward=-0.01,
)
