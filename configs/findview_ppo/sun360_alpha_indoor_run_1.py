_base_ = [
    '../findview_agents/ppo.py',
    '../rl_envs/basic.py',
    '../trainers/base.py'
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
rl_env = dict(
    name='Basic',
    success_reward=100.0,
    slack_reward=-0.01,
    end_type='bell',
    end_type_param=10,
)
base_trainer = dict(
    run_id=1,
    num_envs=16,
    num_updates=5000,
    ckpt_interval=500,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=2500,
)