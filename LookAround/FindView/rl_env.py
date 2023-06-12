#!/usr/bin/env python3

from typing import Callable, Optional, Union

import gymnasium as gym
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
        num_episodes_per_img: int = -1,
        dtype: Union[np.dtype, torch.dtype] = torch.float32,
        device: torch.device = torch.device("cpu"),
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
            num_episodes_per_img=num_episodes_per_img,
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

        # some metrics to follow
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

    @property
    def number_of_episodes(self) -> Optional[int]:
        return self._env.number_of_episodes

    def _reset_metrics(self) -> None:
        self._prev_dist = self._env.get_metrics()["l1_distance"]

    def reset(self):
        observations = self._env.reset()
        self._reset_metrics()
        return observations

    def get_reward_range(self):
        return (-np.inf, np.inf)

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


def build_func(cfg: Config, name: str, registry: Registry, **kwargs):
    rlenv_cls = registry.get(name)
    assert (
        rlenv_cls is not None
    ), f"{name} is not in the {registry.name} registry"

    return rlenv_cls(
        cfg=cfg,
        **kwargs,
    )


# Create a RL Env Registry
RLEnvRegistry = Registry("rlenvs", build_func=build_func)


@RLEnvRegistry.register_module(name="Basic")
class BasicFindviewRLEnv(FindViewRLEnv):
    def __init__(
        self,
        cfg: Config,
        **kwargs,
    ) -> None:
        self._rl_env_cfg = cfg.rl_env
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward

        super().__init__(cfg=cfg, **kwargs)

    def get_reward_range(self) -> tuple:
        # FIXME: better range calculation
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def _end_rewards(self, measures):
        reward_success = 0
        if self._env.episode_over and measures["called_stop"]:
            l1 = measures["l1_distance"]
            reward_success = self._success_reward / (l1 + 1.0)
        elif self._env.episode_over:
            # if agent couldn't finish by the limit, penalize them
            reward_success = -self._success_reward
        return reward_success

    def get_reward(self, observations) -> float:
        # FIXME: make a good reward function here
        measures = self._env.get_metrics()
        curr_dist = measures["l1_distance"]

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
