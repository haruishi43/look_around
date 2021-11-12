_base_ = [
    './inverse_2.py',
]
trainer = dict(
    identifier='adapted_doublelength_max_severity',
    num_updates=30000,
    ckpt_interval=1000,
    log_interval=10,
    pretrained='./pretrained/run_2_doublelength/ckpt.28.pth',
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
corrupter = dict(
    corruptions='all',
    severity=5,
    bounded=False,
    use_clear=True,
)
corruption_scheduler = dict(
    initial_severity=5,
    max_severity=5,
)
