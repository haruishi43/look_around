#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import numpy as np

from mycv.image import imread

from LookAround.config import Config
from LookAround.FindView.equi2pers import Equi2Pers


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


set_seed(0)


def create_sample_episodes_with_single_image(
    num_batches: int,
    num_episodes: Union[int, List[int]],
    img_path: str,
    distance: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Create batches of episodes (at random)
    """
    yaws = np.arange(-179, 181)
    pitch = np.arange(-89, 91)

    if isinstance(num_episodes, int):
        num_episodes = [num_episodes] * num_batches

    assert len(num_episodes) == num_batches, \
        "number of batches and number of episodes doesn't match"

    batches = []
    for b in range(num_batches):
        eps = []
        for _ in range(num_episodes[b]):
            init_yaw = np.random.choice(yaws)
            init_pitch = np.random.choice(pitch)
            while True:
                target_yaw = np.random.choice(yaws)
                target_pitch = np.random.choice(pitch)

                dist = np.sqrt(
                    (target_yaw - init_yaw)**2 + (target_pitch - init_pitch)**2
                )
                # hardcoded distance
                if dist > distance:
                    break
            eps.append(
                dict(
                    img_path=img_path,
                    initial_view=(
                        0,
                        init_pitch,
                        init_yaw,
                    ),
                    target_view=(
                        0,
                        target_pitch,
                        target_yaw,
                    )
                )
            )
        batches.append(eps)

    return batches


class Episode:
    img_path: str
    initial_view: Tuple[int, int, int]
    target_view: Tuple[int, int, int]

    def __init__(
        self,
        img_path: str,
        initial_view: Tuple[int, int, int],
        target_view: Tuple[int, int, int],
    ) -> None:
        assert os.path.exists(img_path), \
            f"ERR: {img_path} is not a valid path"
        self.img_path = img_path
        self.initial_view = initial_view
        self.target_view = target_view

    @property
    def img(self) -> np.ndarray:
        img = imread(self.img_path)
        return img


class SimpleEpisodeIterator:
    def __init__(
        self,
        episodes: List[Dict[str, Any]],
    ) -> None:
        assert len(episodes) > 0, \
            "Number of episodes is zero"
        self.episodes = episodes

        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        episode = self.episodes[self.index]
        self.index += 1
        if self.index >= len(self.episodes):
            self.index = 0
        return episode


class SimpleSingleSimulator:
    equi = None
    episode = None

    def __init__(self) -> None:
        self.equi = None
        self.episode = None

    def load_image(self, img_path) -> None:
        img = imread(img_path)
        self.equi = img

    def get_img(self, chw: bool = True):
        equi = self.equi
        if chw:
            equi = np.transpose(equi, (2, 0, 1))
        return equi


class Agent:
    def __init__(self):
        pass


class SimpleParallelEnv:

    def __init__(
        self,
        batch_size: int,
        episode_iterators: List[SimpleEpisodeIterator],
        configs,  # addict.Dict
    ) -> None:
        self.bs = batch_size
        self.episode_iterators = episode_iterators
        self.equi2pers = Equi2Pers(**configs.equi2pers)
        self.sims = [SimpleSingleSimulator() for _ in range(self.bs)]

    def step(self, action):
        return

    def look_at(
        self,
        rots,
    ) -> np.ndarray:
        imgs = np.stack([sim.get_img() for sim in self.sims], axis=0)
        pers = self.equi2pers(rots, equi=imgs)
        return pers

    def reset(self, i: int):
        ...


if __name__ == "__main__":
    configs = Config.fromfile("configs/sample.py")

    # print(configs.pretty_text)

    img_path = os.path.join(
        configs.dataset.data_root,
        configs.dataset.data_path,
        "indoor",
        "bedroom",
        "pano_azwlwvnimluvvt.jpg",
    )
    assert os.path.exists(img_path), \
        f"ERR: {img_path} is not a valid path"
    batches_of_episodes = create_sample_episodes_with_single_image(
        num_batches=configs.basic.batch_size,
        num_episodes=[2, 3],
        img_path=img_path,
    )

    # episodes = batches_of_episodes[0]
    # for i, episode in enumerate(episodes):
    #     print(i, episode["initial_view"], episode["target_view"])

    batched_episode_iter = [SimpleEpisodeIterator(episodes) for episodes in batches_of_episodes]
    # initialize
    curr_episodes = [next(episode_iter) for episode_iter in batched_episode_iter]

    num_steps = configs.basic.num_steps
    for step in range(num_steps):
        ...
