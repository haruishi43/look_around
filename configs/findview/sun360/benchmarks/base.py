benchmark = dict(
    device='cpu',
    num_envs=1,
    dtype="torch.float32",
    vec_type="threaded",
    video_option=['disk'],
    video_dir="{results_root}/benchmarks/videos/run",
    output_dir="{results_root}/benchmarks/outputs/run",
    log_file="{log_root}/benchmarks_run.log",
    num_eval_episodes=250,
    difficulty="easy",
    bounded=True,
)
