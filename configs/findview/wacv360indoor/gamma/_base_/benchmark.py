_base_ = ['../../../_base_/benchmark.py']
benchmark = dict(
    video_dir=(
        "{results_root}/benchmarks/"
        "findview_{dataset}_{version}/"
        "{bench_name}/"
        "{agent_name}/"
    ),
    metric_path=(
        "{results_root}/benchmarks/"
        "findview_{dataset}_{version}/"
        "{bench_name}/"
        "{agent_name}.json"
    ),
    log_file=(
        "{log_root}/benchmarks/"
        "findview_{dataset}_{version}/"
        "{bench_name}/"
        "{agent_name}.log"
    ),
)
