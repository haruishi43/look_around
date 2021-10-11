#!/usr/bin/env python3

from copy import deepcopy
from functools import partial
import random
from typing import List, Optional, Tuple, Union

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


def filter_out_sub_labels(episode: Episode, sub_labels: Union[List[str], Tuple[str]]) -> bool:
    return episode.sub_label not in sub_labels


def joint_filter_fn(
    episode: Episode,
    names: List[str],
    difficulties: List[str],
) -> bool:
    return (
        filter_by_difficulty(episode, difficulties)
        and filter_by_name(episode, names)
    )


def construct_envs_for_validation(
    cfg: Config,
    num_envs: int,
    split: str,
    is_rlenv: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
    vec_type: str = "threaded",
    difficulty: Optional[str] = None,
    bounded: Optional[bool] = None,
    auto_reset_done: bool = False,
    shuffle: bool = False,
    remove_labels: Optional[Union[List[str], Tuple[str], str]] = "others",
    num_episodes_per_img: int = -1,
) -> VecEnv:
    """Basic initialization of vectorized environments

    It splits the dataset into smaller dataset for each enviornment by first
    splitting the dataset by `img_name`.

    NOTE: VecEnv validation is not consistent when using more than 1 envs and
    `num_eval_episodes=-1`.
    """

    assert split in ("val", "test")

    # make sure that validation cycles since we try to catch when episodes
    # repeat to pause the thread
    cfg.episode_iterator_kwargs.cycle = True

    if remove_labels is not None:
        if isinstance(remove_labels, str):
            remove_labels = [remove_labels]
        assert len(remove_labels) > 0
        filter_fn = partial(filter_out_sub_labels, sub_labels=remove_labels)
    else:
        filter_fn = None

    # 1. preprocessing
    dataset = make_dataset(
        cfg=cfg,
        split=split,
        filter_fn=filter_fn,
    )
    img_names = dataset.get_img_names()

    if len(img_names) > 0:
        if shuffle:
            print("WARNING; shuffling img_names during validation!")
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
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
            num_episodes_per_img=num_episodes_per_img,
            split=split,
            rank=i,
            dtype=dtype,
            device=device,
        )
        env_fn_kwargs.append(kwargs)

    # 3. initialize the vectorized environment
    if vec_type == "mp":
        envs = MPVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    elif vec_type == "equilib":
        # NOTE: faster than multiprocessing
        envs = EquilibVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    elif vec_type == "threaded":
        # NOTE: fastest by far
        envs = ThreadedVecEnv(
            make_env_fn=make_rl_env_fn if is_rlenv else make_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    else:
        raise ValueError(f"ERR: {vec_type} not supported")

    return envs


@RLEnvRegistry.register_module(name="inverse")
class InverseFindViewRLEnv(FindViewRLEnv):

    def __init__(self, cfg: Config, **kwargs) -> None:

        self._rl_env_cfg = cfg.rl_env
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward
        self._param = self._rl_env_cfg.param

        super().__init__(cfg=cfg, **kwargs)

    def get_reward_range(self):
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def _end_rewards(self, measures):

        reward_success = 0
        if self._env.episode_over and measures['called_stop']:
            l1 = measures['l1_distance']
            # l2 = measures['l2_distance']
            reward_success = self._success_reward / (l1 + self._param)
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
        measures = self._env.get_metrics()
        curr_dist = measures['l1_distance']

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


@RLEnvRegistry.register_module(name="bell")
class BellFindViewRLEnv(FindViewRLEnv):

    def __init__(self, cfg: Config, **kwargs) -> None:

        self._rl_env_cfg = cfg.rl_env
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward
        self._param = self._rl_env_cfg.param

        super().__init__(cfg=cfg, **kwargs)

    def get_reward_range(self):
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def _end_rewards(self, measures):

        def bell_curve(x, threshold_steps):
            """Bell curve"""
            # NOTE: when x/threshold_steps == 1, the output is 0.36787944117144233
            return (np.e)**(-(x / threshold_steps)**2)

        reward_success = 0
        if self._env.episode_over and measures['called_stop']:
            l1 = measures['l1_distance']
            # l2 = measures['l2_distance']
            reward_success = self._success_reward * bell_curve(l1, self._param)
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
        measures = self._env.get_metrics()
        curr_dist = measures['l1_distance']

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
