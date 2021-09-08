#!/usr/bin/env python3

"""
NOTE: make registry?
- subclassing makes it complicated

"""

from LookAround.FindView.dataset.episode import Episode
from typing import Optional, Union

import gym
from mycv.utils import Config
import numpy as np
import torch

from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.env import FindViewEnv


class FindViewRLEnv(gym.Env):

    _env: FindViewEnv

    def __init__(
        self,
        cfg: Config,
        **kwargs,
    ) -> None:
        self._cfg = cfg

        self._env = FindViewEnv(
            cfg=cfg,
            **kwargs,
        )

        self.number_of_episodes = self._env.number_of_episodes

        self._min_steps = self._cfg.min_steps
        self._max_steps = self._cfg.max_steps
        self._rl_env_cfg = self._cfg.rl_env_cfgs
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        # self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

        # metrics to follow
        self._prev_dist = None

    @property
    def env(self) -> FindViewEnv:
        return self._env

    @property
    def sim(self) -> FindViewSim:
        return self._env.sim

    @property
    def current_episode(self) -> Episode:
        return self._env.current_episode

    def reset(self):
        observations = self._env.reset()

        self._prev_dist = self._env.get_metrics()['l1_distance_to_target']

        return observations

    def get_reward_range(self):
        # FIXME: better range calculation
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def get_reward(self, observations):
        # FIXME: make a good reward function here

        # Penalty: slack reward for every step it takes
        # value is small
        reward_slack = self._slack_reward

        measures = self._env.get_metrics()
        curr_dist = measures['l1_distance_to_target']

        # Reward/Penalty: if the agent is further from the target, penalize
        # value is either 1 or -1
        reward_dist = self._prev_dist - curr_dist
        self._prev_dist = curr_dist

        # Penality: Looked in the same spot
        # value is in the range of (0 ~ 1)
        num_same_rots = measures['num_same_view']
        reward_same_view = -num_same_rots

        # FIXME: add comparison with shortest path?
        # Reward: when agent hits `stop` and the distance is close
        # giving high reward to calling stop because I think it will be hard
        # to teach an agent to stop...
        reward_success = 0
        if self._env.episode_over and measures['called_stop']:
            if curr_dist < 1:
                reward_success += self._success_reward
            else:
                reward_success -= self._success_reward

        # calculate total rewards
        reward = reward_slack + reward_dist + reward_same_view + reward_success

        return reward

    def get_done(self, observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations):
        return self._env.get_metrics()

    def step_before(self, *args, **kwargs):
        return self._env.step_before(*args, **kwargs)

    def step_after(self):
        observations = self._env.step_after()
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def step(self, *args, **kwargs):
        """Perform an action in the environment.
        :return: :py:`(observations, reward, done, info)`
        """
        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# FIXME: don't have the usecase for this...
def make_rl_env(
    cfg: Config,
    split: str,
    filter_fn=None,
    is_torch: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> FindViewRLEnv:
    if is_torch:
        assert dtype in (torch.float16, torch.float32, torch.float64)
    else:
        assert dtype in (np.float32, np.float64)
    env = FindViewRLEnv(
        cfg=cfg,
        split=split,
        filter_fn=filter_fn,
        is_torch=is_torch,
        dtype=dtype,
        device=device,
    )
    return env