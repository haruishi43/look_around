_base_ = [
    '../../../../_base_/datasets/wacv360indoor/alpha.py',
    '../../../_base_/env.py',
]
sim = dict(
    height=256,
    width=192,
    fov=60.0,
)
