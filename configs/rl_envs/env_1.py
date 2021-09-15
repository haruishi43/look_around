_base_ = [
    '../_base_/datasets/sun360_alpha_indoor.py',
    '../_base_/findview_sim.py'
]
episode_generator_kwargs = dict(
    shuffle=True,
    num_repeat_pseudo=-1,
)
episode_iterator_kwargs = dict(
    cycle=False,
    shuffle=True,
    num_episode_sample=-1,
)
num_envs = 8
rl_env_cfgs = dict(
    success_reward=100.0,
    slack_reward=-0.01,
    end_type="bell",
    end_type_param=10,
)
