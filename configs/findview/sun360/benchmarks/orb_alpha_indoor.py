_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/orb.py',
    './base.py',
]
benchmark = dict(
    device='cpu',
    dtype="np.float32",
)
