_base_ = [
    '../defaults.py'
]
dataset = dict(
    name='sun360',
    version='alpha',
    category='indoor',
    split_ratios=[0.8, 0.1, 0.1],
    num_easy=30,
    num_medium=40,
    num_hard=30,
    fov=90.0,
    min_steps=10,
    max_steps=5000,
    step_size=1,
    pitch_threshold=60,
    max_seconds=10000000,
    json_path="{root}/{name}/{version}/{category}/{split}.json",
    mu=0.0,
    sigma=0.3,
    sample_limit=100000,
)
