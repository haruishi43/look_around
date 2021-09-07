#!/usr/bin/env python3

"""Simple script to checkout a dataset

Dataset
- creates an iterator for val/test
- creates a generator (sub-class of iterator) for train

Why:
- Validation and test sets are static
- Training can be dynamic (create more training set by sampling)
- Training dataset is huge (around 500mb in raw json)
- Allocating all episodes is none sense
- Generators use almost no space at the cost of time
- Two modes during training `Dataset` (static and dynamic)

NOTE:
- `generator` is a sub-class of `iterator`, but we can't really define `__len__()`
  since there are no iterable data class inside; meaning it is a concept where
  an `iterator` can keep on creating infinite data unless stopped.
- In our implementation, we call it DynamicIterator (compared with StaticIterator)

"""


import argparse
import os

from tqdm import tqdm

from mycv.utils.config import Config

from LookAround.FindView.dataset.static_dataset import StaticDataset
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset
from LookAround.FindView.dataset.sampling import DifficultySampler


from scripts.helpers import func_timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    cfg = Config.fromfile(config_path)
    print(">>> Config:")
    print(cfg.pretty_text)

    # basic params
    sun360_root = os.path.join(cfg.data_root, cfg.dataset_name)
    dataset_root = os.path.join(cfg.dataset_root, cfg.dataset_name, cfg.version, cfg.category)

    # some checks before moving on
    assert os.path.exists(sun360_root)
    assert os.path.exists(dataset_root)

    # params
    split = "train"

    print(f">>> Checking dataset for {split}")

    if split in ['val', 'test']:
        dataset = StaticDataset(cfg=cfg, split=split)

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
        for i in tqdm(range(len(dataset))):
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
        for i in tqdm(range(2 * len(dataset))):
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

        @func_timer
        def get_under_hard(episodes):
            new_episodes = []
            for episode in episodes:
                if episode.difficulty in ['easy', 'medium']:
                    new_episodes.append(episode)
            return new_episodes

        episode_iterator.filter_episodes(filter_func=get_under_hard)

        # NOTE: need to cycle first so self._iterator is set
        for i in tqdm(range(len(dataset))):
            episode = next(episode_iterator)
        for i in range(50):
            episode = next(episode_iterator)
            print(episode.img_name, episode.episode_id, episode.difficulty)

    elif split == "train":
        dataset = DynamicDataset(cfg=cfg)
        sampler = DifficultySampler(
            difficulty='easy',
            fov=cfg.fov,
            min_steps=cfg.min_steps,
            max_steps=cfg.max_steps,
            step_size=cfg.step_size,
            threshold=cfg.pitch_threshold,
        )

        # test for generator
        print(">>> Testing generator")

        print(">>> Test #1")
        episode_generator = dataset.get_generator(
            sampler=sampler,
            shuffle=False,
            num_repeat_pseudo=-1,
        )
        num_iter = 100
        for i in range(num_iter):
            episode = next(episode_generator)
            print(i, episode.path, episode.difficulty)

        print(">>> Test #2")
        episode_generator = dataset.get_generator(
            sampler=sampler,
            shuffle=True,
            num_repeat_pseudo=-1,
        )
        num_iter = 10000
        for i in range(num_iter):
            episode = next(episode_generator)
            print(i, episode.img_name, episode.difficulty)

        # change difficulty
        print(">>> Test #3: Changed Diff")
        sampler.set_difficulty('hard')
        for i in range(num_iter):
            episode = next(episode_generator)
            print(i, episode.img_name, episode.difficulty)

    else:
        raise ValueError(f"{split} is not a valid split")
