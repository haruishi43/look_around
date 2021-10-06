base_trainer = dict(
    run_id=9999,
    num_envs=8,
    num_updates=7500,
    num_ckpts=-1,
    ckpt_interval=500,
    total_num_steps=-1.0,
    log_interval=10,
    video_option=["disk"],
    ckpt_dir="{results_root}/checkpoints/run_{run_id}",
    video_dir="{results_root}/videos/run_{run_id}",
    tb_dir="{tb_root}/run_{run_id}",
    log_file="{log_root}/{split}_run_{run_id}.log",
    verbose=True,
)
scheduler = dict(
    initial_difficulty='easy',
    difficulties=('easy', 'medium', 'hard'),
    update_interval=2500,
)
train = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    resume=False,
)
val = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_eval_episodes=250,
)
test = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    num_eval_episodes=250,
    difficulty="easy",
    bounded=True,
)