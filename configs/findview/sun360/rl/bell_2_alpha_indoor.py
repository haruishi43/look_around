_base_ = [
    '../agents/ppo.py',
    '../rl_envs/bell.py',
    '../trainers/base.py',
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
rl_env = dict(
    success_reward=10.0,
    param=5,
)
trainer = dict(
    run_id=2,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
