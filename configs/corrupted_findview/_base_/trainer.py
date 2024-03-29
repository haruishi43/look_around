trainer = dict(
    run_id=999999,
    identifier=None,
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_envs=16,
    num_updates=7500,
    num_ckpts=-1,
    ckpt_interval=500,
    total_num_steps=-1.0,
    log_interval=10,
    ckpt_dir=(
        "{results_root}/checkpoints/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "run_{run_id}"
    ),
    tb_dir=(
        "{tb_root}/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "run_{run_id}"
    ),
    log_file=(
        "{log_root}/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}.log"
    ),
    resume=False,
    pretrained=None,
    verbose=True,
)
validator = dict(
    num_envs=16,
    num_episodes=-1,
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    video_option=["disk"],
    save_metrics=True,
    video_dir=(
        "{results_root}/videos/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    metric_dir=(
        "{results_root}/metrics/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    tb_dir=(
        "{tb_root}/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    log_file=(
        "{log_root}/"
        "corrupted_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}.log"
    ),
    severity=3,
    difficulty="hard",
    bounded=False,
    remove_labels="others",
    num_episodes_per_img=1,
)
scheduler = dict(
    initial_difficulty="easy",
    difficulties=("easy", "medium", "hard"),
    update_interval=2500,
)
corruption_scheduler = dict(
    initial_severity=0,
    max_severity=5,
    update_interval=1500,
)
