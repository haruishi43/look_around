_base_ = [
    '../_base_/defaults.py',
    '../../../agents/ppo.py',
    '../_base_/benchmark.py',
]
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
