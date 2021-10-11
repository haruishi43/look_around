_base_ = [
    '../agents/ppo.py',
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
    end_type='inverse',
    end_type_param=10,
)
trainer = dict(
    run_id=20,
    num_envs=8,
    num_updates=30000,
    ckpt_interval=1000,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=10000,
)