#!/usr/bin/env python3

"""Naive Vectorized Environment for RL

exploit equi2pers's batch sampling

"""

from copy import deepcopy
from functools import partial
import random
from typing import List, Type, Union

from mycv.utils import Config
import numpy as np
import torch

from LookAround.FindView.dataset import Episode, PseudoEpisode, make_dataset
from LookAround.FindView.env import FindViewEnv, FindViewRLEnv
from LookAround.FindView.sim import batch_sample


class VecEnv(object):

    envs: List[Union[FindViewEnv, FindViewEnv]]

    def __init__(
        self,
        make_env_fn,
        env_fn_kwargs,
    ) -> None:

        self._num_envs = len(env_fn_kwargs)

        # initialize envs
        self.envs = [
            make_env_fn(**env_fn_kwargs[i])
            for i in range(self._num_envs)
        ]

    @property
    def num_envs(self):
        """number of individual environments.
        """
        return self._num_envs

    def reset(self):
        batch_obs = []
        for env in self.envs:
            batch_obs.append(env.reset())
        return batch_obs

    def step(self, actions: List[str]):
        batch_ret = []

        # get rotations
        rots = []
        for env, action in zip(self.envs, actions):
            rot = env.step_before(action)
            rots.append(rot)

        # NOTE: really hacky way of batch sampling
        sims = [env.sim for env in self.envs]
        batch_sample(sims, rots)

        # make sure to get observations
        for env in self.envs:
            batch_ret = env.step_after()

        return batch_ret

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def filter_by_name(
    episode: Union[Episode, PseudoEpisode],
    names: List[str],
) -> bool:
    return episode.img_name in names


def filter_by_sub_labels(
    episode: Union[Episode, PseudoEpisode],
    sub_labels: List[str],
) -> bool:
    return episode.sub_label in sub_labels


def make_env_fn(
    cfg: Config,
    env_cls: Type[Union[FindViewEnv, FindViewRLEnv]],
    filter_fn,
    split: str,
    rank: int,
    is_torch: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> Union[FindViewEnv, FindViewRLEnv]:

    env = env_cls(
        cfg=cfg,
        split=split,
        filter_fn=filter_fn,
        is_torch=is_torch,
        dtype=dtype,
        device=device,
    )
    env.seed(rank)

    return env


def construct_envs(
    cfg: Config,
    env_cls: Type[Union[FindViewEnv, FindViewRLEnv]],
    split: str,
    is_torch: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> VecEnv:

    num_envs = cfg.num_envs

    # get all dataset
    dataset = make_dataset(cfg=cfg, split=split)

    # FIXME: maybe use sub_labels too?
    img_names = dataset.get_img_names()

    if len(img_names) > 0:
        random.shuffle(img_names)

        assert len(img_names) >= num_envs, (
            "reduce the number of environments as there "
            "aren't enough diversity in images"
        )

    print(img_names)

    img_name_splits = [[] for _ in range(num_envs)]
    for idx, img_name in enumerate(img_names):
        img_name_splits[idx % len(img_name_splits)].append(img_name)

    assert sum(map(len, img_name_splits)) == len(img_names)

    env_fn_kwargs = []
    for i in range(num_envs):

        _cfg = deepcopy(cfg)  # make sure to clone
        _cfg.seed = i  # iterator and sampler depends on this

        # FIXME: maybe change how the devices are allocated
        # if there are multiple devices (cuda), it would be
        # nice to somehow split the devices evenly

        kwargs = dict(
            cfg=_cfg,
            env_cls=env_cls,
            filter_fn=partial(filter_by_name, names=img_name_splits[i]),
            split=split,
            rank=i,
            is_torch=is_torch,
            dtype=dtype,
            device=device,
        )
        env_fn_kwargs.append(kwargs)

    envs = VecEnv(
        make_env_fn=make_env_fn,
        env_fn_kwargs=env_fn_kwargs,
    )
    return envs
