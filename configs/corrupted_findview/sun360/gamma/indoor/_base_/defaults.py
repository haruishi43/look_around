_base_ = [
    '../../../../../_base_/datasets/sun360/gamma_indoor.py',
    '../../../../_base_/env.py',
]
sim = dict(
    height=192,
    width=256,
    fov=60.0,
)
