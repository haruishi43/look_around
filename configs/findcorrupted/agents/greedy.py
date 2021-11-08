_base_ = [
    '../_base_/benchmark.py',
]
greedy = dict(
    chance=0.001,
    seed=0,
)
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
