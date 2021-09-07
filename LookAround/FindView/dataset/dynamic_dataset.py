#!/usr/bin/env python3

"""Dynamic Dataset

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

from .episode import Episode, PseudoEpisode
from .sampling import Sampler

T = TypeVar("T", bound=PseudoEpisode)


class DynamicDataset(Generic[T]):
    """Dynamic Dataset

    """
    pseudos: List[T]

    def __init__(
        self,
        cfg: Config,
    ) -> None:

        self.pseudos = []

        dataset_json_path = cfg.dataset_json_path.format(
            root=cfg.dataset_root,
            name=cfg.dataset_name,
            version=cfg.version,
            category=cfg.category,
            split="train",
        )
        data_dir = os.path.join(cfg.data_root, cfg.dataset_name)
        assert os.path.exists(dataset_json_path), \
            f"ERR: {dataset_json_path} doesn't exist"
        assert os.path.exists(data_dir), \
            f"ERR: {data_dir} doesn't exist"

        with open(dataset_json_path, "r") as f:
            self.from_json(f.read(), data_dir=data_dir)

    def __len__(self) -> int:
        return len(self.pseudos)

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
            pseudo = PseudoEpisode(**d)
            self.pseudos.append(pseudo)

    def get_generator(
        self,
        sampler: Sampler,
        **kwargs,
    ) -> "DynamicGenerator":
        return DynamicGenerator(
            self.pseudos,
            sampler=sampler,
            **kwargs,
        )

    def filter_episodes(self, filter_fn: Callable[[T], bool]) -> "DynamicDataset":
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        assert len(new_episodes) > 0, \
            "ERR: filtered all episodes; no episode for dataset"
        new_dataset = copy(self)
        new_dataset.episodes = new_episodes
        return new_dataset


class DynamicGenerator(Iterator):

    def __init__(
        self,
        pseudos: Sequence[T],
        sampler: Sampler,
        shuffle: bool = True,
        num_repeat_pseudo: int = -1,
        seed: int = None,
    ) -> None:

        if seed:
            # FIXME: is this necessary?
            random.seed(seed)
            np.random.seed(seed)

        if not isinstance(pseudos, list):
            pseudos = list(pseudos)

        self.sampler = sampler
        self.shuffle = shuffle
        self.num_repeat_pseudo = num_repeat_pseudo

        # shuffle
        if self.shuffle:
            random.shuffle(pseudos)

        self.pseudos = pseudos
        self._pseudo_iterator = iter(self.pseudos)
        self.current_pseudo = next(self._pseudo_iterator)
        self._repeated = 0

    def __iter__(self) -> "DynamicGenerator":
        return self

    def __next__(self) -> Episode:
        """The main logic for handling how episodes will be iterated.
        """
        pseudo = self.get_pseudo()
        episode = self.sampler(pseudo)
        return episode

    def get_pseudo(self) -> PseudoEpisode:
        if self._repeated >= self.num_repeat_pseudo:
            # change pseudo
            pseudo = next(self._pseudo_iterator, None)

            if pseudo is None:
                pseudos = deepcopy(self.pseudos)
                if self.shuffle:
                    random.shuffle(pseudos)
                self._pseudo_iterator = iter(pseudos)
                pseudo = next(self._pseudo_iterator)

            self.current_pseudo = pseudo
            self._repeated = 0
        else:
            # repeat
            pseudo = self.current_pseudo
            self._repeated += 1
        return pseudo
