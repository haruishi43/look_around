_base_ = [
    '../_base_/defaults.py'
]
dataset = dict(
    name='sun360',
    version='test',
    category='indoor',
    split_ratios=[0.5, 0.25, 0.25],
    num_easy=3,
    num_medium=4,
    num_hard=3,
    fov=90.0,
    min_steps=10,
    max_steps=2000,
    step_size=1,
    pitch_threshold=60,
    max_seconds=10000000,
    json_path="{root}/{name}/{version}/{category}/{split}.json",
    mu=0.0,
    sigma=0.3,
    sample_limit=100000,
)
