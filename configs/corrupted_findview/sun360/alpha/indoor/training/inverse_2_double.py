_base_ = [
    './inverse_2.py',
]
trainer = dict(
    identifier='double',
    num_envs=32,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
)
validator = dict(
    num_envs=16,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=2000,
)
