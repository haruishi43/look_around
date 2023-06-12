_base_ = ["../defaults.py"]
dataset = dict(
    split_ratios=[0.8, 0.1, 0.1],
    num_easy=2,
    num_medium=4,
    num_hard=4,
    difficulty="easy",
    bounded=False,
    fov=90.0,
    min_steps=10,
    max_steps=5000,
    step_size=1,
    pitch_threshold=60,
    max_seconds=10000000,
    mu=0.0,
    sigma=0.3,
    sample_limit=100000,
)
