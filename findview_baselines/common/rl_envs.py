#!/usr/bin/env python3

from copy import deepcopy
from functools import partial
import random
from typing import List, Optional, Union

import numpy as np
import torch

from LookAround.config import Config
from LookAround.FindView.dataset import Episode, make_dataset
from LookAround.FindView.rl_env import FindViewRLEnv, RLEnvRegistry
from LookAround.FindView.vec_env import (
    EquilibVecEnv,
    MPVecEnv,
    ThreadedVecEnv,
    VecEnv,
    filter_by_difficulty,
    filter_by_name,
    make_env_fn,
    make_rl_env_fn,
)


def joint_filter_fn(
    episode: Episode,
    names: List[str],
    difficulties: List[str],
) -> bool:
    return filter_by_difficulty(episode, difficulties) and filter_by_name(episode, names)


def construct_envs_for_validation(
    cfg: Config,
    split: str,
    is_rlenv: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
    vec_type: str = "threaded",
    difficulty: Optional[str] = None,
    bounded: Optional[bool] = None,
) -> VecEnv:
    """Basic initialization of vectorized environments

    It splits the dataset into smaller dataset for each enviornment by first
    splitting the dataset by `img_name`.
    """

    # 1. preprocessing
    # NOTE: make sure to seed first so that we get consistant tests
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    num_envs = cfg.base_trainer.num_envs

    # get all dataset
    assert split in ("val", "test")
    dataset = make_dataset(cfg=cfg, split=split)
    img_names = dataset.get_img_names()

    if len(img_names) > 0:
        # NOTE: uses global random state
        random.shuffle(img_names)

        assert len(img_names) >= num_envs, (
            "reduce the number of environments as there "
            "aren't enough diversity in images"
        )

    img_name_splits = [[] for _ in range(num_envs)]
    for idx, img_name in enumerate(img_names):
        img_name_splits[idx % len(img_name_splits)].append(img_name)

    assert sum(map(len, img_name_splits)) == len(img_names)

    if difficulty is None:
        difficulty = cfg.dataset.difficulty
    if bounded is None:
        bounded = cfg.dataset.bounded
    if bounded:
        difficulties = (difficulty,)
    else:
        if difficulty == "easy":
            difficulties = (difficulty,)
        elif difficulty == "medium":
            difficulties = ("easy", "medium")
        elif difficulty == "hard":
            difficulties = ("easy", "medium", "hard")

    for diff in difficulties:
        assert diff in ("easy", "medium", "hard")

    # 2. create initialization arguments for each environment
    env_fn_kwargs = []
    for i in range(num_envs):

        _cfg = Config(deepcopy(cfg))  # make sure to clone
        _cfg.seed = _cfg.seed + i  # iterator and sampler depends on this

        # print(">>>", i)
        # print(_cfg.pretty_text)
        # print(len(img_name_splits[i]), len(img_names))

        # FIXME: maybe change how the devices are allocated
        # if there are multiple devices (cuda), it would be
        # nice to somehow split the devices evenly

        kwargs = dict(
            cfg=_cfg,
            filter_fn=partial(  # NOTE: filter by both difficulty and names
                joint_filter_fn,
                names=img_name_splits[i],
                difficulties=difficulties,
            ),
            split=split,
            rank=i,
            dtype=dtype,
            device=device,
        )
        env_fn_kwargs.append(kwargs)

    # 3. initialize the vectorized environment
    if vec_type == "mp":
        # FIXME: Very slow. Torch using multi-thread?
        envs = MPVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
        )
    elif vec_type == "equilib":
        # NOTE: faster than multiprocessing
        envs = EquilibVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
        )
    elif vec_type == "threaded":
        # NOTE: fastest by far
        envs = ThreadedVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
        )
    else:
        raise ValueError(f"ERR: {vec_type} not supported")

    return envs


@RLEnvRegistry.register_module(name="Env1")
class FindViewRLEnv1(FindViewRLEnv):

    # FIXME: same as Basic for now

    def __init__(self, cfg: Config, **kwargs) -> None:

        self._rl_env_cfg = cfg.rl_env_cfgs
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward
        self._end_reward_type = self._rl_env_cfg.end_type
        self._end_reward_param = self._rl_env_cfg.end_type_param

        super().__init__(cfg=cfg, **kwargs)

    def get_reward_range(self):
        # FIXME: better range calculation
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def _reset_metrics(self) -> None:
        self._prev_dist = self._env.get_metrics()['l1_distance_to_target']

    def _end_rewards(self, measures):

        def bell_curve(x, threshold_steps):
            """Bell curve"""
            # NOTE: when x/threshold_steps == 1, the output is 0.36787944117144233
            return (np.e)**(-(x / threshold_steps)**2)

        reward_success = 0
        if self._env.episode_over and measures['called_stop']:
            l1 = measures['l1_distance_to_target']
            # l2 = measures['l2_distance_to_target']

            # FIXME: is success reward too high???
            # runs 1, 2:
            # reward_success = self._success_reward - l1
            # run 3:
            if self._end_reward_type == "inverse":
                reward_success = self._success_reward / (l1 + self._end_reward_param)
            # run 4: threshold = 10
            elif self._end_reward_type == "bell":
                reward_success = self._success_reward * bell_curve(l1, self._end_reward_param)  # FIXME parametrize
            else:
                raise ValueError("Reward function parameter is not set correctly")

        elif self._env.episode_over:
            # if agent couldn't finish by the limit, penalize them
            reward_success = -self._success_reward

        return reward_success

    def _same_view_penalty(self, measures):
        # Penality: Looked in the same spot
        # value is in the range of (0 ~ 1)
        num_same_rots = measures['num_same_view']
        reward_same_view = -num_same_rots
        return reward_same_view

    def get_reward(self, observations):
        # FIXME: make a good reward function here
        measures = self._env.get_metrics()
        curr_dist = measures['l1_distance_to_target']

        # Penalty: slack reward for every step it takes
        # value is small
        reward_slack = self._slack_reward

        # Reward/Penalty: if the agent is further from the target, penalize
        # value is either 1 or -1
        coef_dist = 0.1  # run_2=0.01, run1=1.0, etc...
        reward_dist = coef_dist * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        reward_success = self._end_rewards(measures)

        reward = reward_slack + reward_dist + reward_success

        return reward
