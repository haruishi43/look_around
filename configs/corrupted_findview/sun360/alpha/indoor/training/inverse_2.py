_base_ = [
    "./inverse_1.py",
]
dataset = dict(
    difficulty="easy",
    bounded=False,
)
rl_env = dict(
    param=10.0,
)
trainer = dict(
    run_id=2,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty="easy",
    update_interval=5000,
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=2000,
)
