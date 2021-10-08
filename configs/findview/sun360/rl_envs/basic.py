_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py'
]
rl_env = dict(
    name="Basic",
    success_reward=100.0,
    slack_reward=-0.01,
    end_type="bell",
    end_type_param=10,
)
