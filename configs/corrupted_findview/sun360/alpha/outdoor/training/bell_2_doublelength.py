_base_ = [
    "./bell_2.py",
]
trainer = dict(
    run_id=2,
    identifier="doublelength",
    num_updates=30000,
    ckpt_interval=1000,
    log_interval=10,
)
scheduler = dict(
    initial_difficulty="easy",
    update_interval=10000,
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=5000,
)
