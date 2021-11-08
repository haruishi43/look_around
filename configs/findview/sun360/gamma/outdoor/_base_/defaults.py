_base_ = [
    '../../../../../_base_/datasets/sun360/gamma_outdoor.py',
    '../../../../_base_/env.py',
]
sim = dict(
    height=360,
    width=480,
    fov=60.0,
)
