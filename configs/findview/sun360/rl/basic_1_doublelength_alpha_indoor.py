_base_ = [
    './basic_1_alpha_indoor.py',
]
trainer = dict(
    identifier='half',
    num_envs=16,
    num_updates=30000,
    ckpt_interval=1000,
    log_interval=10,
)
validator = dict(
    num_envs=16,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=10000,
)
