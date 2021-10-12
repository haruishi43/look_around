_base_ = [
    '../agents/ppo.py',
    '../rl_envs/basic.py',
    '../trainers/base.py',
]
dataset = dict(
    difficulty='easy',
    bounded=False,
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
