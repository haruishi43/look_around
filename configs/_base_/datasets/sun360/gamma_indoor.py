_base_ = ['../dataset.py']
dataset = dict(
    name='sun360',
    version='gamma',
    category='indoor',
    split_ratios=[0.8, 0.1, 0.1],
    num_easy=2,
    num_medium=4,
    num_hard=4,
    fov=60.0,
    json_path=(
        "{root}/"
        "{name}/"
        "{version}/"
        "{category}/"
        "{split}.json"
    ),
)
