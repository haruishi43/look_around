_base_ = [
    "../_base_/datasets/sun360/alpha_indoor.py",
]

sim = dict(
    height=128,
    width=128,
    fov=90.0,
    sampling_mode="bilinear",
)
episode_generator_kwargs = dict(
    shuffle=True,
    num_repeat_pseudo=-1,
)
episode_iterator_kwargs = dict(
    cycle=False,
    shuffle=False,
    num_episode_sample=-1,
)

benchmark = dict(
    device="cpu",
    dtype="torch.float32",
    num_threads=4,
    video_dir=(
        "{results_root}/benchmarks/"
        "findview_{dataset}_{version}_{category}/"
        "{bench_name}/"
        "{agent_name}/"
    ),
    metric_path=(
        "{results_root}/benchmarks/"
        "findview_{dataset}_{version}_{category}/"
        "{bench_name}/"
        "{agent_name}.json"
    ),
    log_file=(
        "{log_root}/benchmarks/"
        "findview_{dataset}_{version}_{category}/"
        "{bench_name}/"
        "{agent_name}.log"
    ),
    num_episodes=-1,
    difficulty="hard",
    bounded=True,
    remove_labels="others",
    num_episodes_per_img=1,
    save_video=True,
    beautify=True,
)

policy = dict(
    action_distribution_type="categorical",
    use_log_std=False,
    use_softplus=False,
    min_std=1e-6,
    max_std=1,
    min_log_std=-5,
    max_log_std=2,
    action_activation="tanh",
)
ppo = dict(
    clip_param=0.2,
    ppo_epoch=4,
    num_mini_batch=1,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    lr=2.5e-4,
    eps=1e-5,
    max_grad_norm=0.5,
    num_steps=128,
    use_gae=True,
    use_linear_lr_decay=True,
    use_linear_clip_decay=True,
    gamma=0.99,
    tau=0.95,
    reward_window_size=50,
    use_normalized_advantage=False,
    hidden_size=512,
    use_double_buffered_sampler=False,
)

rl_env = dict(
    name="inverse",
    success_reward=100.0,
    slack_reward=-0.01,
    param=10.0,
)
dataset = dict(
    difficulty="easy",
    bounded=False,
)
trainer = dict(
    run_id=2,
    identifier=None,
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_envs=16,
    num_updates=15000,
    num_ckpts=-1,
    ckpt_interval=500,
    total_num_steps=-1.0,
    log_interval=10,
    ckpt_dir=(
        "{results_root}/checkpoints/"
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "run_{run_id}"
    ),
    tb_dir=(
        "{tb_root}/"
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "run_{run_id}"
    ),
    log_file=(
        "{log_root}/"
        "findview_{dataset}_{version}_{category}/"
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
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    metric_dir=(
        "{results_root}/metrics/"
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    tb_dir=(
        "{tb_root}/"
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}"
    ),
    log_file=(
        "{log_root}/"
        "findview_{dataset}_{version}_{category}/"
        "{rlenv}/"
        "{split}/"
        "run_{run_id}.log"
    ),
    difficulty="hard",
    bounded=False,
    remove_labels="others",
    num_episodes_per_img=1,
)
scheduler = dict(
    initial_difficulty="easy",
    difficulties=("easy", "medium", "hard"),
    update_interval=5000,
)
