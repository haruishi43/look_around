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
