_base_ = [
    '../findview/sun360/agents/ppo.py',
    '../findview/sun360/rl_envs/basic.py',
    '../findview/sun360/trainers/base.py',
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
ppo = dict(
    ppo_epoch=4,
    num_mini_batch=2,
    num_steps=128,
)
trainer = dict(
    run_id=999999,
    identifier="test_mini",
    vec_type="threaded",
    num_envs=2,
    num_updates=1000,
    ckpt_interval=500,
    log_interval=10,
    resume=False,
    pretrained=None,
)
validator = dict(
    num_envs=2,
    num_episodes=-1,
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    difficulty="easy",
    bounded=False,
    remove_labels="others",
    num_episodes_per_img=1,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=-1,
)
