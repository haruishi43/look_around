_base_ = [
    "../../../../_base_/datasets/wacv360indoor/gamma.py",
    "../../../_base_/env.py",
]
sim = dict(
    height=192,
    width=256,
    fov=60.0,
)
