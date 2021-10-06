#!/usr/bin/env python3

"""Testing Static and Dynamic Dataset

"""

import os

from LookAround.config import Config
from LookAround.FindView.dataset.static_dataset import StaticDataset
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset
from LookAround.FindView.dataset.sampling import DifficultySampler


def test_static_dataset():

    data_dir = "tests/data/sun360"
    dataset_json_path = "tests/dataset/sun360/test/indoor/test.json"
    fov = 90.0
    min_steps = 10
    max_steps = 2000
    step_size = 1
    pitch_threshold = 60
    max_seconds = 10000000
    seed = 1

    dataset = StaticDataset(
        data_dir=data_dir,
        dataset_json_path=dataset_json_path,
        fov=fov,
        min_steps=min_steps,
        max_steps=max_steps,
        step_size=step_size,
        pitch_threshold=pitch_threshold,
        max_seconds=max_seconds,
        seed=seed,
    )

    assert fov == dataset.fov
    assert min_steps == dataset.min_steps
    assert max_steps == dataset.max_steps
    assert step_size == dataset.step_size
    assert max_seconds == dataset.max_seconds
    assert pitch_threshold == dataset.pitch_threshold

    print(len(dataset))
    print(dataset.get_img_names())
    print(dataset.get_sub_labels())


def test_dynamic_dataset():

    data_dir = "tests/data/sun360"
    dataset_json_path = "tests/dataset/sun360/test/indoor/train.json"
    fov = 90.0
    min_steps = 10
    max_steps = 2000
    step_size = 1
    pitch_threshold = 60
    max_seconds = 10000000
    seed = 1
    mu = 0.0
    sigma = 0.3
    sample_limit = 100000

    dataset = DynamicDataset(
        data_dir=data_dir,
        dataset_json_path=dataset_json_path,
        fov=fov,
        min_steps=min_steps,
        max_steps=max_steps,
        step_size=step_size,
        pitch_threshold=pitch_threshold,
        max_seconds=max_seconds,
        seed=seed,
        mu=mu,
        sigma=sigma,
        sample_limit=sample_limit,
    )

    assert fov == dataset.fov
    assert min_steps == dataset.min_steps
    assert max_steps == dataset.max_steps
    assert step_size == dataset.step_size
    assert max_seconds == dataset.max_seconds
    assert pitch_threshold == dataset.pitch_threshold

    print(dataset.get_img_names())
    print(dataset.get_sub_labels())

    assert isinstance(dataset.sampler, DifficultySampler)


def test_iterator():

    cfg_path = os.path.join("tests/configs/datasets", "sun360_indoor.py")
    assert os.path.exists(cfg_path)
    cfg = Config.fromfile(cfg_path)

    dataset = StaticDataset.from_config(
        cfg=cfg,
        split="test",
    )

    assert len(dataset) > 0

    # test for iterator
    print(">>> Testing iterator")

    # test if the dataset runs in order
    print(">>> Test #1")
    episode_iterator = dataset.get_iterator(
        cycle=False,
        shuffle=False,
        num_episode_sample=-1,
    )
    for i in range(len(dataset)):
        episode = next(episode_iterator)
        assert i == episode.episode_id
        print(episode.path, episode.episode_id)

    # test run cycle
    print(">>> Test #2")
    episode_iterator = dataset.get_iterator(
        cycle=True,
        shuffle=False,
        num_episode_sample=-1,
    )
    for i in range(2 * len(dataset)):
        episode = next(episode_iterator)
        assert i % len(dataset) == episode.episode_id

    # test random
    print(">>> Test #3")
    episode_iterator = dataset.get_iterator(
        cycle=True,
        shuffle=True,
        num_episode_sample=-1,
    )
    for i in range(10):
        episode = next(episode_iterator)
        print(episode.img_name, episode.episode_id, episode.difficulty)

    # test filtering episodes
    print(">>> Test #4")

    # @func_timer
    def get_under_hard(episodes):
        new_episodes = []
        for episode in episodes:
            if episode.difficulty in ['easy', 'medium']:
                new_episodes.append(episode)
        return new_episodes

    episode_iterator.filter_episodes(filter_func=get_under_hard)

    # NOTE: need to cycle first so self._iterator is set
    for i in range(len(dataset)):
        episode = next(episode_iterator)
    for i in range(50):
        episode = next(episode_iterator)
        print(episode.img_name, episode.episode_id, episode.difficulty)


def test_generator():

    cfg_path = os.path.join("tests/configs/datasets", "sun360_indoor.py")
    assert os.path.exists(cfg_path)
    cfg = Config.fromfile(cfg_path)

    dataset = DynamicDataset.from_config(
        cfg=cfg,
        split="train",
    )

    sampler = dataset.sampler  # NOTE: alias

    # test for generator
    print(">>> Testing generator")

    print(">>> Test #1")
    episode_generator = dataset.get_generator(
        shuffle=False,
        num_repeat_pseudo=-1,
    )
    num_iter = 100
    for i in range(num_iter):
        episode = next(episode_generator)
        print(i, episode.path, episode.difficulty)

    print(">>> Test #2")
    episode_generator = dataset.get_generator(
        shuffle=True,
        num_repeat_pseudo=-1,
    )
    num_iter = 100
    for i in range(num_iter):
        episode = next(episode_generator)
        print(i, episode.img_name, episode.difficulty)

    # change difficulty
    print(">>> Test #3: Changed Diff")
    sampler.set_difficulty('hard')
    for i in range(num_iter):
        episode = next(episode_generator)
        print(i, episode.img_name, episode.difficulty)
