#!/usr/bin/env python3

from typing import Callable, Optional, Union

import gym
import numpy as np
import torch

from LookAround.config import Config
from LookAround.core import Registry
from LookAround.FindView.dataset.episode import Episode
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.env import FindViewEnv

__all__ = [
    "FindViewRLEnv",
    "BasicFindviewRLEnv",
    "RLEnvRegistry",
]


class FindViewRLEnv(gym.Env):

    # Hidden Properties
    _env: FindViewEnv

    def __init__(
        self,
        cfg: Config,
        split: str,
        filter_fn: Optional[Callable[..., bool]] = None,
        dtype: Union[np.dtype, torch.dtype] = torch.float32,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """FindView RL Environment

        NOTE: initialize using root Config

        params:
        - cfg (Config)
        - split (str)
        - filter_fn (Callable): None
        - dtype (np.dtype, torch.dtype): torch.float32
        - device (torch.device): 'cpu'
        """

        self._env = FindViewEnv.from_config(
            cfg=cfg,
            split=split,
            filter_fn=filter_fn,
            dtype=dtype,
            device=device,
        )

        dataset = self._env._dataset

        # from dataset
        self._min_steps = dataset.min_steps
        self._max_steps = dataset.max_steps

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.reward_range = self.get_reward_range()

    @property
    def env(self) -> FindViewEnv:
        return self._env

    @property
    def sim(self) -> FindViewSim:
        return self._env.sim

    @property
    def current_episode(self) -> Episode:
        return self._env.current_episode

    @property
    def number_of_episodes(self) -> Optional[int]:
        return self._env.number_of_episodes

    def _reset_metrics(self) -> None:
        # NOTE: it's a `gym` thing, don't really need to implement
        # return (-np.inf, np.inf)
        raise NotImplementedError

    def reset(self):
        observations = self._env.reset()
        self._reset_metrics()
        return observations

    def get_reward_range(self):
        raise NotImplementedError

    def get_reward(self, observations) -> float:
        raise NotImplementedError

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
        """Perform an action in the environment
        return (observations, reward, done, info)
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

    def change_difficulty(self, difficulty: str, bounded: bool) -> None:
        self._env.change_difficulty(difficulty=difficulty, bounded=bounded)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def build_func(name: str, registry: Registry, cfg: Config, **kwargs):
    rlenv_cls = registry.get(name)
    assert rlenv_cls is not None, \
        f"{name} is not in the {registry.name} registry"

    return rlenv_cls(
        cfg=cfg,
        **kwargs,
    )


# Create a RL Env Registry
RLEnvRegistry = Registry('rlenvs', build_func=build_func)


@RLEnvRegistry.register_module(name='Basic')
class BasicFindviewRLEnv(FindViewRLEnv):

    def __init__(
        self,
        cfg: Config,
        **kwargs,
    ) -> None:

        # rl spectific variables
        self._rl_env_cfg = cfg.rl_env
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward
        self._end_reward_type = self._rl_env_cfg.end_type
        self._end_reward_param = self._rl_env_cfg.end_type_param

        # Intitialize parent
        super().__init__(
            cfg=cfg,
            **kwargs,
        )

        # metrics to follow
        self._prev_dist = None

    def get_reward_range(self) -> tuple:
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

    def get_reward(self, observations) -> float:
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
