_base_ = [
    '../_base_/benchmark.py',
]
sm = dict(
    action="right",
)
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
