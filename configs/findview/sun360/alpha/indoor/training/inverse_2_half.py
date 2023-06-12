_base_ = [
    "./inverse_2.py",
]
trainer = dict(
    identifier="half",
    num_envs=8,
    num_updates=30000,
    ckpt_interval=1000,
    log_interval=10,
)
validator = dict(
    num_envs=8,
)
scheduler = dict(
    initial_difficulty="easy",
    update_interval=10000,
)
