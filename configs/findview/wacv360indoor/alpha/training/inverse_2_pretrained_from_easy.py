_base_ = [
    './inverse_2.py',
]
trainer = dict(
    identifier='adapted_easy',
    num_envs=16,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
    pretrained='./pretrained/run_2_doublelength/ckpt.28.pth',
)
validator = dict(
    num_envs=16,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
