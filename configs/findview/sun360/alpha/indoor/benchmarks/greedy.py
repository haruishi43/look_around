_base_ = [
    "../_base_/defaults.py",
    "../../../../agents/greedy.py",
    "../../../../_base_/benchmark.py",
]
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
