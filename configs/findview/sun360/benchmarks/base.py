benchmark = dict(
    device='cpu',
    num_envs=1,
    dtype="torch.float32",
    vec_type="threaded",
    video_option=['disk'],
    video_dir=(
        "{results_root}/benchmarks/"
        "{agent}/"
        "videos/"
    ),
    output_dir="{results_root}/benchmarks/outputs/run",
    log_file="{log_root}/benchmarks_run.log",
    num_eval_episodes=-1,
    difficulty="easy",
    bounded=True,
)
