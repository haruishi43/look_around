_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/greedy.py',
    './base.py',
]
benchmark = dict(
    device=0,
    num_envs=1,
    dtype="torch.float32",
    vec_type="threaded",
    video_dir="{results_root}/benchmarks/videos/greedy",
    output_dir="{results_root}/benchmarks/outputs/greedy",
    log_file="{log_root}/benchmarks_greedy.log",
)
