_base_ = [
    '../agents/ppo.py',
    '../rl_envs/inverse_outdoor.py',
    '../trainers/base.py',
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
rl_env = dict(
    param=10.0,
)
trainer = dict(
    run_id=1,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
