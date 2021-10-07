#!/usr/bin/env python3

from copy import copy, deepcopy
import os
import json
import random
from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Sequence,
    TypeVar,
)

import numpy as np

from LookAround.config import Config
from LookAround.FindView.dataset.episode import Episode

T = TypeVar("T", bound=Episode)


class StaticDataset(Generic[T]):

    episodes: List[T]

    _fov: float
    _min_steps: int
    _max_steps: int
    _step_size: int
    _max_seconds: int
    _seed: int

    def __init__(
        self,
        data_dir: os.PathLike,
        dataset_json_path: os.PathLike,
        fov: float,
        min_steps: int = 10,
        max_steps: int = 5000,
        step_size: int = 1,
        pitch_threshold: int = 60,
        max_seconds: int = 10000000,
        seed: int = 0,
    ) -> None:
        """Static Dataset

        params:
        - data_dir (PathLike): data root `data`
        - dataset_json_path (PathLike): json file of the dataset
        - fov (float)
        - min_steps (int)
        - max_steps (int)
        - step_size (int)
        - max_seconds (int)
        - pitch_threshold (int)
        - seed (int)

        Pre-defined in the way that the Dataset (episodes) are already
        created and made into a `.json` file with ALL attributes (of the
        `Episode`) filled out.

        - This class is mainly used to get an iterator (Val, Test)
        - Sometimes, it might be useful for Training when we want less
        time for sampling new dataset in `DynamicDataset`

        """

        assert os.path.exists(dataset_json_path), \
            f"ERR: {dataset_json_path} doesn't exist"
        assert os.path.exists(data_dir), \
            f"ERR: {data_dir} doesn't exist"

        self.episodes = []

        # Keep parameters
        # this is needed by the simulator and env
        self._fov = fov
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._step_size = step_size
        self._max_seconds = max_seconds
        self._pitch_threshold = pitch_threshold
        self._seed = seed

        with open(dataset_json_path, "r") as f:
            self.episodes_from_json(f.read(), data_dir=data_dir)

    @classmethod
    def from_config(cls, cfg: Config, split: str = "test"):
        """Initialize StaticDataset using Config

        params:
        - cfg (Config)
        - split (str): split referes to the json file name
        """

        data_dir = os.path.join(cfg.data_root, cfg.dataset.name)
        dataset_json_path = cfg.dataset.json_path.format(
            root=cfg.dataset_root,
            name=cfg.dataset.name,
            version=cfg.dataset.version,
            category=cfg.dataset.category,
            split=split,
        )

        return cls(
            data_dir=data_dir,
            dataset_json_path=dataset_json_path,
            fov=cfg.dataset.fov,
            min_steps=cfg.dataset.min_steps,
            max_steps=cfg.dataset.max_steps,
            step_size=cfg.dataset.step_size,
            pitch_threshold=cfg.dataset.pitch_threshold,
            max_seconds=cfg.dataset.max_seconds,
            seed=cfg.seed,  # NOTE: seed not `cfg.dataset`
        )

    @property
    def fov(self) -> float:
        assert self._fov is not None
        return self._fov

    @property
    def min_steps(self) -> int:
        assert self._min_steps is not None
        return self._min_steps

    @property
    def max_steps(self) -> int:
        assert self._max_steps is not None
        return self._max_steps

    @property
    def step_size(self) -> int:
        assert self._step_size is not None
        return self._step_size

    @property
    def max_seconds(self) -> int:
        assert self._max_seconds is not None
        return self._max_seconds

    @property
    def pitch_threshold(self) -> int:
        assert self._pitch_threshold is not None
        return self._pitch_threshold

    def __len__(self) -> int:
        return len(self.episodes)

    def get_img_names(self) -> List[str]:
        return list(set([e.img_name for e in self.episodes]))

    def get_sub_labels(self) -> List[str]:
        return list(set([e.sub_label for e in self.episodes]))

    def episodes_from_json(
        self,
        json_str: str,
        data_dir: os.PathLike,
    ) -> None:
        """Load json data to Episodes
        """
        deserialized = json.loads(json_str)
        assert len(deserialized) > 0 and isinstance(deserialized, list), \
            "ERR: contents of json string unreadable"

        for d in deserialized:
            # make sure that full path is loaded
            d["path"] = os.path.join(os.getcwd(), data_dir, d["path"])
            # FIXME: do we need this check before hand?
            assert os.path.exists(d["path"]), \
                f"ERR: {d['path']} doesn't exist"
            episode = Episode(**d)
            self.episodes.append(episode)

    def get_iterator(
        self,
        **kwargs,
    ) -> "StaticIterator":
        return StaticIterator(
            self.episodes,
            **kwargs,
        )

    def filter_dataset(self, filter_fn: Callable[[T], bool]) -> "StaticDataset":
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        assert len(new_episodes) > 0, \
            "ERR: filtered all episodes; no episode for dataset"
        new_dataset = copy(self)  # copies all attributes
        new_dataset.episodes = new_episodes
        return new_dataset


class StaticIterator(Iterator):

    def __init__(
        self,
        episodes: Sequence[T],
        cycle: bool = False,
        shuffle: bool = False,
        num_episode_sample: int = -1,
        seed: int = 0,
    ) -> None:

        # FIXME: is this necessary?
        # making it multi-thread safe
        self.rst = random.Random(seed)
        self.np_rst = np.random.RandomState(seed)

        # sample episodes
        if num_episode_sample >= 0:
            episodes = self.np_rst.choice(
                episodes, num_episode_sample, replace=False
            )

        if not isinstance(episodes, list):
            episodes = list(episodes)

        self.all_episodes = deepcopy(episodes)
        self.cycle = cycle
        self.shuffle = shuffle

        # shuffle
        if self.shuffle:
            self.rst.shuffle(episodes)

        # set iterater
        self.episodes = episodes
        self._iterator = iter(self.episodes)

    def __iter__(self) -> "StaticIterator":
        return self

    def __next__(self) -> Episode:
        """The main logic for handling how episodes will be iterated.
        """
        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration

            episodes = deepcopy(self.episodes)

            # filter here?
            # filtering every cycle might not be efficient

            # shuffle
            if self.shuffle:
                self.rst.shuffle(episodes)

            self._iterator = iter(episodes)
            next_episode = next(self._iterator)

        return next_episode

    def _reset_episodes(self) -> None:
        self.episodes = deepcopy(self.all_episodes)

    def filter_episodes(self, filter_func) -> None:
        """Call this function during training when you want to edit episodes
        on-the-fly (i.e., increase difficulty)
        """
        # first get all episodes
        episodes = deepcopy(self.all_episodes)
        episodes = filter_func(episodes)
        assert len(episodes) > 0, isinstance(episodes, list)
        self.episodes = episodes
        # hopefully when `next()` is called, new episodes are called in
        # making an iterator directly is faster (i.e. `filter`), but
        # no guarantee of `__len__`
