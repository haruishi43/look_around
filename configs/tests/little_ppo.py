_base_ = [
    '../findview/sun360/agents/ppo.py',
    '../findview/sun360/rl_envs/basic.py',
    '../findview/sun360/trainers/base.py',
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
rl_env = dict(
    name='Basic',
    success_reward=100.0,
    slack_reward=-0.01,
    end_type='bell',
    end_type_param=10,
)
trainer = dict(
    run_id=999998,
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_envs=8,
    num_updates=1000,
    ckpt_interval=10,
    log_interval=10,
    ckpt_dir="{results_root}/checkpoints/test_run_{run_id}",
    video_dir="{results_root}/videos/test_run_{run_id}",
    tb_dir="{tb_root}/test_run_{run_id}",
    log_file="{log_root}/{split}_test_run_{run_id}.log",
    resume=False,
    pretrained=None,
)
validator = dict(
    num_eval_episodes=250,
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    difficulty="easy",
    bounded=False,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=-1,
)
