_base_ = [
    "../_base_/defaults.py",
    "../../../../agents/orb.py",
    "../../../../_base_/benchmark.py",
]
benchmark = dict(
    device="cpu",
    dtype="np.float32",
)
