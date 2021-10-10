_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/ppo.py',
    './base.py',
]
benchmark = dict(
    device=0,
    num_envs=1,
    dtype="torch.float32",
    vec_type="threaded",
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    video_dir="{results_root}/benchmarks/videos/ppo",
    output_dir="{results_root}/benchmarks/outputs/ppo",
    log_file="{log_root}/benchmarks_ppo.log",
)
