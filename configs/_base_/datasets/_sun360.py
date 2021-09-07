_base_ = [
    '../defaults.py'
]
dataset_name = 'sun360'
split_ratios = [0.8, 0.1, 0.1]
num_easy = 30
num_medium = 40
num_hard = 30
fov = 90.0
pitch_threshold = 60
min_steps = 10
max_steps = 5000
max_seconds = 10000000
step_size = 1
dataset_json_path = "{root}/{name}/{version}/{category}/{split}.json"
sim = dict(
    height=256,
    width=256,
    fov=90.0,
    sampling_mode="bilinear",
)
episode_generator_kwargs = dict(
    shuffle=True,
    num_repeat_pseudo=-1,
)
episode_iterator_kwargs = dict(
    cycle=False,
    shuffle=False,
    num_episode_sample=-1,
)
