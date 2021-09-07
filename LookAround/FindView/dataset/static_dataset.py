#!/usr/bin/env python3

"""Static Dataset

Purpose:
- Dataset is in charge of creating `iterable`s
- Each item in the `iterable` is an instance of `Episode`
-

Why use `Generic` and `TypeVar`?
- We want `Dataset` to be able to use various `Episode`s
- Meaning any subclass of `Episode` is allowed for `Dataset

"""

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

from mycv.utils.config import Config
import numpy as np

from .episode import Episode

T = TypeVar("T", bound=Episode)


class StaticDataset(Generic[T]):
    """Static Dataset

    Pre-defined in the way that the Dataset (episodes) are already
    created and made into a `.json` file with ALL attributes (of the
    `Episode`) filled out.

    - This class is mainly used to get an iterator (Val, Test)
    - Sometimes, it might be useful for Training when we want less
      time for sampling new dataset in `DynamicDataset`

    """
    episodes: List[T]

    def __init__(
        self,
        cfg: Config,
        split: str,
    ) -> None:

        # NOTE: only using `Config` for initializing `episodes`
        # initialize episodes
        self.episodes = []

        # NOTE: `split` should be `val` or `test` (sometimes `train`)
        dataset_json_path = cfg.dataset_json_path.format(
            root=cfg.dataset_root,
            name=cfg.dataset_name,
            version=cfg.version,
            category=cfg.category,
            split=split,
        )
        data_dir = os.path.join(cfg.data_root, cfg.dataset_name)
        assert os.path.exists(dataset_json_path), \
            f"ERR: {dataset_json_path} doesn't exist"
        assert os.path.exists(data_dir), \
            f"ERR: {data_dir} doesn't exist"

        with open(dataset_json_path, "r") as f:
            self.from_json(f.read(), data_dir=data_dir)

    def __len__(self) -> int:
        return len(self.episodes)

    def get_img_names(self) -> List[str]:
        return list(set([e.img_name for e in self.episodes]))

    def get_sub_labels(self) -> List[str]:
        return list(set([e.sub_label for e in self.episodes]))

    def from_json(
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
        return StaticIterator(self.episodes, **kwargs)

    def filter_dataset(self, filter_fn: Callable[[T], bool]) -> "StaticDataset":
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        assert len(new_episodes) > 0, \
            "ERR: filtered all episodes; no episode for dataset"
        new_dataset = copy(self)
        new_dataset.episodes = new_episodes
        return new_dataset


class StaticIterator(Iterator):

    def __init__(
        self,
        episodes: Sequence[T],
        cycle: bool = False,
        shuffle: bool = False,
        num_episode_sample: int = -1,
        seed: int = None,
    ) -> None:

        if seed:
            # FIXME: is this necessary?
            random.seed(seed)
            np.random.seed(seed)

        # sample episodes
        if num_episode_sample >= 0:
            episodes = np.random.choice(
                episodes, num_episode_sample, replace=False
            )

        if not isinstance(episodes, list):
            episodes = list(episodes)

        self.all_episodes = deepcopy(episodes)
        self.cycle = cycle
        self.shuffle = shuffle

        # shuffle
        if self.shuffle:
            random.shuffle(episodes)

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
                random.shuffle(episodes)

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
