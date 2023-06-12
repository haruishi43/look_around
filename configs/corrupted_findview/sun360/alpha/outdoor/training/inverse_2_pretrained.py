_base_ = [
    "./inverse_2.py",
]
trainer = dict(
    identifier="adapted",
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
    pretrained="./pretrained/run_2/ckpt.best.pth",
)
scheduler = dict(
    initial_difficulty="easy",
    update_interval=2500,
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=2500,
)
