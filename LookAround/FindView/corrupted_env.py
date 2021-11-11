#!/usr/bin/env python3

import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from LookAround.config import Config
from LookAround.core.improc import (
    post_process_for_render,
    post_process_for_render_torch,
    to_tensor,
)
from LookAround.FindView.dataset import make_dataset
from LookAround.FindView.env import FindViewEnv
from LookAround.FindView.sim import FindViewSim, Tensor
from LookAround.FindView.corruptions import corrupt, get_corruption_names


class CorruptionModule(object):

    severity_levels = (0, 1, 2, 3, 4, 5)

    _corruptions: List[str]
    _severity: int
    _bounded: bool

    def __init__(
        self,
        corruptions: Union[List[str], Tuple[str], str],
        severity: int = 0,
        bounded: bool = True,
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:

        assert severity in self.severity_levels, \
            f"ERR: given severity is {severity}"
        self._severity = severity
        self._bounded = bounded

        _names = get_corruption_names(subset='all')

        if isinstance(corruptions, str):
            corruptions = [corruptions]
        assert len(corruptions) > 0
        for corruption in corruptions:
            assert corruption in _names
        self._corruptions = corruptions
        self.deterministic = deterministic
        self.seed_value = seed
        self.seed(seed)

    @classmethod
    def from_config(cls, cfg: Config):

        corruptions = cfg.corrupter.corruptions
        if isinstance(corruptions, str):
            if corruptions in ('all', 'blur', 'noise', 'digital', 'weather'):
                corruptions = get_corruption_names(corruptions)
            elif corruptions in get_corruption_names('all'):
                corruptions = [corruptions]
            else:
                raise ValueError()
        else:
            assert len(corruptions) > 0
            corruptions = corruptions

        return cls(
            corruptions=corruptions,
            severity=cfg.corrupter.severity,
            bounded=cfg.corrupter.bounded,
            seed=cfg.seed,
            deterministic=cfg.corrupter.deterministic,
        )

    @property
    def corruption_list(self) -> str:
        return self._corruptions

    @property
    def severity(self) -> int:
        assert self._severity is not None
        return self._severity

    @severity.setter
    def severity(self, level: int):
        assert level in self.severity_levels, \
            f"ERR: given severity is {level}"
        self._severity = level

    def corrupt(
        self,
        img: Tensor,
        name: Optional[str] = None,
    ) -> Tuple[Tensor, str, int]:

        if (name is not None) and isinstance(name, str):
            assert name in self._corruptions, \
                f"ERR: {name} is no in {self._corruptions}"
        else:
            name = self.rst.choice(self._corruptions)

        if self._bounded:
            severity = self._severity
        else:
            # FIXME: we assume that severities are indexed in order from 0
            severities = self.severity_levels[:self._severity + 1]
            severity = self.rst.choice(severities)

        # convert to rgb numpy hwc
        if torch.is_tensor(img):
            is_torch = True
            device = img.device
            dtype = img.dtype
            assert dtype in (torch.float16, torch.float32, torch.float64)
            img = post_process_for_render_torch(
                img=img,
                to_bgr=False,
            )
        else:
            is_torch = False
            dtype = img.dtype
            assert dtype in (np.float32, np.float64)
            img = post_process_for_render(
                img=img,
                to_bgr=False,
            )

        if self.deterministic:
            state = np.random.get_state()
            np.random.seed(self.seed_value)

        corrupted = corrupt(
            image=img,
            severity=severity,
            corruption_name=name,
        )  # output is np.uint8

        if self.deterministic:
            np.random.set_state(state)

        # NOTE: assume target type is float
        # convert to chw and float
        if is_torch:
            corrupted = to_tensor(corrupted)
            corrupted = corrupted.type(dtype)
            corrupted = corrupted.to(device)
        else:
            corrupted = np.transpose(corrupted, (2, 0, 1))
            corrupted = corrupted / 255.0
            corrupted = corrupted.astype(dtype)

        return corrupted, name, severity

    def seed(self, seed: int) -> None:
        # multi-thread safe
        self.rst = random.Random(seed)
        self.np_rst = np.random.RandomState(seed)


class CorruptedFindViewEnv(FindViewEnv):

    # Modules
    _corruptor: CorruptionModule

    # Hidden Properties
    _corruption_name: str
    _corruption_severity: int

    def __init__(
        self,
        corrupter: CorruptionModule,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._corruptor = corrupter

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        split: str,
        filter_fn: Optional[Callable[..., bool]] = None,
        num_episodes_per_img: int = -1,
        dtype: Union[np.dtype, torch.dtype] = torch.float32,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialization from Config
        """

        # Initialize dataset
        dataset = make_dataset(
            cfg=cfg,
            split=split,
            filter_fn=filter_fn,
            num_episodes_per_img=num_episodes_per_img,
        )

        # Initialize episode iterator
        if split in ('train'):
            iter_options = cfg.episode_generator_kwargs
            iter_options['seed'] = cfg.seed
            episode_iterator = dataset.get_generator(**iter_options)
        elif split in ('val', 'test'):
            iter_options = cfg.episode_iterator_kwargs
            iter_options['seed'] = cfg.seed
            episode_iterator = dataset.get_iterator(**iter_options)
        else:
            raise ValueError(f"got {split} as split which is unsupported")

        # Initialize simulator
        sim = FindViewSim.from_config(cfg)
        sim.inititialize_loader(dtype=dtype, device=device)

        # Initialize corrupter
        corrupter = CorruptionModule.from_config(cfg)

        return cls(
            corrupter=corrupter,
            dataset=dataset,
            episode_iterator=episode_iterator,
            sim=sim,
            seed=cfg.seed,
        )

    def get_info(self) -> dict:
        info = super().get_info()
        assert self._corruption_name is not None
        assert self._corruption_severity is not None
        info['corruption'] = self._corruption_name
        info['severity'] = self._corruption_severity
        return info

    def reset(self, name: Optional[str] = None) -> Dict[str, Tensor]:
        self._reset_stats()

        self._current_episode = next(self._episode_iterator)
        assert self._current_episode is not None, "ERR: called reset, but there are no more episodes in the iterator"

        initial_rotation = self._current_episode.initial_rotation
        target_rotation = self._current_episode.target_rotation
        episode_path = self._current_episode.path

        self._rot_tracker.reset(initial_rotation)
        pers, _target = self._sim.reset(
            equi_path=episode_path,
            initial_rotation=initial_rotation,
            target_rotation=target_rotation,
        )

        (
            corrupted, corruption_name, corruption_severity
        ) = self._corruptor.corrupt(
            img=_target,
            name=name,
        )
        self._sim.target = corrupted  # NOTE: make sure `corrupted` is copied
        self._corruption_name = corruption_name
        self._corruption_severity = corruption_severity

        return self._make_base_obs(
            pers=pers,
            target=self._sim.target,  # NOTE: make sure `corrupted` is copied
        )

    @property
    def get_corruption_names(self) -> List[str]:
        return self._corruptor._corruptions

    def change_corruption_levels(
        self,
        severity: int,
    ) -> None:
        assert severity in self._corruptor.severity_levels, \
            f"ERR: {severity} is not in {self._corruptor.severity_levels}"
        self._corruptor.severity = severity
