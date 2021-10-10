_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    './base.py',
]
benchmark = dict(
    device=0,
    num_envs=1,
    dtype="torch.float32",
    vec_type="threaded",
    video_dir="{results_root}/benchmarks/videos/human",
    output_dir="{results_root}/benchmarks/outputs/human",
    log_file="{log_root}/benchmarks_human.log",
)
