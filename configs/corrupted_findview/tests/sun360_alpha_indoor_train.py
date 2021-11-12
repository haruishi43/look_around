_base_ = [
    '../sun360/alpha/indoor/training/inverse_2_half.py',
]
trainer = dict(
    identifier='half_test',
    num_envs=8,
    num_updates=120,
    ckpt_interval=60,
    log_interval=10,
    pretrained='./pretrained/run_2_doublelength/ckpt.28.pth',
)
validator = dict(
    num_envs=8,
)
scheduler = dict(
    initial_difficulty='hard',
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=20,
)
