_base_ = [
    './inverse_2.py',
]
trainer = dict(
    identifier='adapted',
    num_envs=32,
    num_updates=10000,
    ckpt_interval=500,
    log_interval=10,
    pretrained='./pretrained/run_2_doublelength/ckpt.28.pth',
)
validator = dict(
    num_envs=16,
)
scheduler = dict(
    initial_difficulty='hard',
    update_interval=5000,
)
