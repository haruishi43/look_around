_base_ = ["./sim.py"]
episode_generator_kwargs = dict(
    shuffle=True,
    num_repeat_pseudo=-1,
)
episode_iterator_kwargs = dict(
    cycle=False,
    shuffle=False,
    num_episode_sample=-1,
)
corrupter = dict(
    corruptions='all',
    severity=3,
    bounded=True,
    deterministic=True,
)
