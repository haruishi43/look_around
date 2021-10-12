_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/ppo.py',
    './base.py',
]
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
