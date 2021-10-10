_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/feature_matching.py',
    './base.py',
]
benchmark = dict(
    device='cpu',
    num_envs=1,
    dtype="np.float32",
    vec_type="threaded",
    video_dir="{results_root}/benchmarks/videos/fm",
    output_dir="{results_root}/benchmarks/outputs/fm",
    log_file="{log_root}/benchmarks_fm.log",
)
