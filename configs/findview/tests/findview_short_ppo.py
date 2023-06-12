_base_ = [
    "../agents/ppo.py",
    "../findview/sun360/alpha/indoor/rl_envs/basic.py",
    "../findview/sun360/_base_/trainer.py",
]
dataset = dict(
    difficulty="easy",
    bounded=False,
)
trainer = dict(
    run_id=999999,
    identifier="test_short",
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_envs=16,
    num_updates=200,
    ckpt_interval=50,
    log_interval=10,
    resume=False,
    pretrained=None,
)
validator = dict(
    num_envs=16,
    num_episodes=10,
    ckpt_path="ckpt.best.pth",
    difficulty="easy",
    bounded=False,
)
scheduler = dict(
    initial_difficulty="easy",
    update_interval=10,
)
