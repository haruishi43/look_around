_base_ = [
    "../../../../../_base_/datasets/sun360/alpha_outdoor.py",
    "../../../../_base_/env.py",
]
corrupter = dict(
    corruptions="all",
    severity=0,
    bounded=False,
    use_clear=True,
    deterministic=False,
)
