_base_ = [
    '../../../../../_base_/datasets/sun360/gamma_outdoor.py',
    '../../../../_base_/env.py',
]
sim = dict(
    height=256,
    width=192,
    fov=60.0,
)
