_base_ = [
    '../../../_base_/datasets/sun360_alpha_indoor.py',
    '../../../_base_/envs/findview.py',
    '../agents/orb.py',
    './base.py',
]
fm = dict(
    feature_type="ORB",
)
benchmark = dict(
    device='cpu',
    dtype="np.float32",
)
