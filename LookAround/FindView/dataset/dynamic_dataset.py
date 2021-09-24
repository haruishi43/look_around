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
from LookAround.FindView.dataset.episode import Episode, PseudoEpisode
from LookAround.FindView.dataset.sampling import DifficultySampler

T = TypeVar("T", bound=PseudoEpisode)


class DynamicDataset(Generic[T]):

    pseudos: List[T]

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
        max_seconds: int = 10000000,
        pitch_threshold: int = 60,
        seed: int = 0,
        mu: float = 0.0,
        sigma: float = 0.3,
        sample_limit: int = 100000,
    ) -> None:
        """Dynamic Dataset

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
        - mu (float)
        - sigma (float)
        - sample_limit (int)
        """

        assert os.path.exists(dataset_json_path), \
            f"ERR: {dataset_json_path} doesn't exist"
        assert os.path.exists(data_dir), \
            f"ERR: {data_dir} doesn't exist"

        self.pseudos = []

        # Keep parameters
        # this is needed by the simulator and env
        self._fov = fov
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._step_size = step_size
        self._max_seconds = max_seconds
        self._pitch_threshold = pitch_threshold
        self._seed = seed

        # Initialize sampler here
        # FIXME: add functionality to choose the sampler type
        # currently there are no need since DifficultySampler is the only
        # type of sampler
        self.sampler = DifficultySampler(
            difficulty="easy",  # NOTE: default difficulty is temporary easy
            fov=self._fov,
            min_steps=self._min_steps,
            max_steps=self._max_steps,
            step_size=self._step_size,
            threshold=self._pitch_threshold,
            seed=self._seed,
            mu=mu,
            sigma=sigma,
            num_tries=sample_limit,
        )

        with open(dataset_json_path, "r") as f:
            self.pseudos_from_json(f.read(), data_dir=data_dir)

    @classmethod
    def from_config(cls, cfg: Config, split: str = "train"):
        """Initialize DynamicDataset using Config

        params:
        - cfg (Config)
        - split (str): split refers to json file name
        """

        dataset_json_path = cfg.dataset.json_path.format(
            root=cfg.dataset_root,
            name=cfg.dataset.name,
            version=cfg.dataset.version,
            category=cfg.dataset.category,
            split=split,
        )
        data_dir = os.path.join(cfg.data_root, cfg.dataset.name)

        return cls(
            data_dir=data_dir,
            dataset_json_path=dataset_json_path,
            fov=cfg.dataset.fov,
            min_steps=cfg.dataset.min_steps,
            max_steps=cfg.dataset.max_steps,
            step_size=cfg.dataset.step_size,
            max_seconds=cfg.dataset.max_seconds,
            pitch_threshold=cfg.dataset.pitch_threshold,
            seed=cfg.seed,  # NOTE: seed not `cfg.dataset`
            mu=cfg.dataset.mu,
            sigma=cfg.dataset.sigma,
            sample_limit=cfg.dataset.sample_limit,
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
        return len(self.pseudos)

    def get_img_names(self) -> List[str]:
        return list(set([p.img_name for p in self.pseudos]))

    def get_sub_labels(self) -> List[str]:
        return list(set([p.sub_label for p in self.pseudos]))

    def pseudos_from_json(
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
        **kwargs,
    ) -> "DynamicGenerator":
        return DynamicGenerator(
            self.pseudos,
            sampler=self.sampler,  # NOTE: passing reference to generator
            **kwargs,
        )

    def filter_dataset(self, filter_fn: Callable[[T], bool]) -> "DynamicDataset":
        new_pseudos = []
        for pseudo in self.pseudos:
            if filter_fn(pseudo):
                new_pseudos.append(pseudo)
        assert len(new_pseudos) > 0, \
            "ERR: filtered all pseudos; no pseudo for dataset"
        new_dataset = copy(self)  # copies all attributes
        new_dataset.pseudos = new_pseudos
        return new_dataset


class DynamicGenerator(Iterator):

    def __init__(
        self,
        pseudos: Sequence[T],
        sampler: DifficultySampler,  # FIXME: only support `DifficultySampler` for now
        shuffle: bool = True,
        num_repeat_pseudo: int = -1,
        seed: int = 0,
    ) -> None:

        # FIXME: is this necessary?
        self.rst = random.Random(seed)
        self.np_rst = np.random.RandomState(seed)

        if not isinstance(pseudos, list):
            pseudos = list(pseudos)

        self.sampler = sampler
        self.shuffle = shuffle
        self.num_repeat_pseudo = num_repeat_pseudo

        # shuffle
        if self.shuffle:
            self.rst.shuffle(pseudos)

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
                    self.rst.shuffle(pseudos)
                self._pseudo_iterator = iter(pseudos)
                pseudo = next(self._pseudo_iterator)

            self.current_pseudo = pseudo
            self._repeated = 0
        else:
            # repeat
            pseudo = self.current_pseudo
            self._repeated += 1
        return pseudo

    def set_difficulty(self, difficulty: str):
        """Set Difficulty inside the sampler"""

        # FIXME: the difficulties are somewhat confusing
        # - easy -> easy
        # - medium -> easy and medium
        # - hard -> easy, medium, and hard
        self.sampler.set_difficulty(difficulty=difficulty)
