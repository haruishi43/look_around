#!/usr/bin/env python3

"""
Control agent training and evaluation

- Communication with simulators
- Observations (obs)
- Rewards
- Action
- Dataset
- Episode Scheduler

- Vectorized Env uses this

"""

import random
import time

from LookAround.FindView.dataset.episode import Episode
from typing import Dict, List, Optional, Union

import gym
from gym import spaces
from mycv.utils import Config
import numpy as np
import torch

from LookAround.FindView.metric import (
    count_same_rots,
    distance_to_target,
)

from LookAround.core.spaces import ActionSpace, EmptySpace
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.dataset import make_dataset
from LookAround.FindView.dataset.static_dataset import StaticDataset, StaticIterator
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset, DynamicGenerator
from LookAround.FindView.dataset.sampling import DifficultySampler, Sampler


class Actions:
    all = ["up", "down", "right", "left", "stop"]
    UP: str = "up"
    DOWN: str = "down"
    RIGHT: str = "right"
    LEFT: str = "left"
    STOP: str = "stop"


FindViewActions = Actions()


class RotationTracker(object):

    rot: Dict[str, int]
    history: List[Dict[str, int]]

    def __init__(
        self,
        inc: int = 1,
        pitch_threshold: int = 60,
    ) -> None:
        self.inc = inc
        self.pitch_threshold = pitch_threshold

        self.rot = None
        self.history = []

    def initialize(self, initial_rotation) -> None:
        self.rot = initial_rotation
        self.history = []

    def convert(self, action: str) -> Dict[str, float]:
        assert self.rot is not None
        pitch = self.rot['pitch']
        yaw = self.rot['yaw']

        if action == FindViewActions.UP:
            pitch += self.inc

        elif action == FindViewActions.DOWN:
            pitch -= self.inc

        elif action == FindViewActions.RIGHT:
            yaw += self.inc

        elif action == FindViewActions.LEFT:
            yaw -= self.inc

        if pitch >= self.pitch_threshold:
            pitch = self.pitch_threshold
        elif pitch <= -self.pitch_threshold:
            pitch = -self.pitch_threshold
        if yaw > 180:
            yaw -= 2 * 180
        elif yaw <= -180:
            yaw += 2 * 180

        rot = {
            "roll": 0,
            "pitch": int(pitch),
            "yaw": int(yaw),
        }

        self.rot = rot
        self.history.append(rot)

        return rot


class FindViewEnv(object):

    _dataset: Union[DynamicDataset, StaticDataset]
    _sampler: Optional[Sampler]
    _episode_iterator: Union[DynamicGenerator, StaticIterator]
    _rot_tracker: RotationTracker
    _sim: FindViewSim

    _current_episode: Optional[Episode]
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self,
        cfg: Config,
        split='train',
        filter_fn=None,
        is_torch=True,
        dtype=torch.float32,
        device=torch.device('cpu'),
    ) -> None:

        self._cfg = cfg

        # initialize dataset
        self._dataset = make_dataset(
            cfg=self._cfg,
            split=split,
            filter_fn=filter_fn,
        )
        if split == 'train':
            self._sampler = DifficultySampler(
                difficulty='easy',  # difficulty...
                fov=self._cfg.fov,
                min_steps=self._cfg.min_steps,
                max_steps=self._cfg.max_steps,
                step_size=self._cfg.step_size,
                threshold=self._cfg.pitch_threshold,
                seed=self._cfg.seed,
                num_tries=100000,  # num tries is pretty large
            )
            iter_options = self._cfg.episode_generator_kwargs
            iter_options['seed'] = self._cfg.seed
            self._episode_iterator = self._dataset.get_generator(
                sampler=self._sampler,
                **iter_options,
            )
        else:
            self._sampler = None
            iter_options = self._cfg.episode_iterator_kwargs
            iter_options['seed'] = self._cfg.seed
            self._episode_iterator = self._dataset.get_iterator(
                **iter_options,
            )

        # initialize simulator
        self._sim = FindViewSim(
            **self._cfg.sim,
        )
        self._sim.inititialize_loader(
            is_torch=is_torch,
            dtype=dtype,
            device=device,
        )

        # initialize rotation tracker
        self._rot_tracker = RotationTracker(
            inc=self._cfg.step_size,
            pitch_threshold=self._cfg.pitch_threshold,
        )

        # gym spaces here?
        self.action_space = ActionSpace(
            {
                action_name: EmptySpace()
                for action_name in FindViewActions.all
            }
        )
        if is_torch:
            self.observation_space = spaces.Dict(
                {
                    "pers": spaces.Box(
                        low=torch.finfo(dtype).min,
                        high=torch.finfo(dtype).max,
                        shape=(
                            self._cfg.sim['height'],
                            self._cfg.sim['width'],
                            3,
                        ),
                    ),
                    "target": spaces.Box(
                        low=torch.finfo(dtype).min,
                        high=torch.finfo(dtype).max,
                        shape=(
                            self._cfg.sim['height'],
                            self._cfg.sim['width'],
                            3,
                        ),
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "pers": spaces.Box(
                        low=np.finfo(dtype).min,
                        high=np.finfo(dtype).max,
                        shape=(
                            self._cfg.sim['height'],
                            self._cfg.sim['width'],
                            3,
                        ),
                    ),
                    "target": spaces.Box(
                        low=np.finfo(dtype).min,
                        high=np.finfo(dtype).max,
                        shape=(
                            self._cfg.sim['height'],
                            self._cfg.sim['width'],
                            3,
                        ),
                    ),
                }
            )

        # initialize stuff
        self._max_episode_seconds = self._cfg.max_seconds
        self._max_episode_steps = self._cfg.max_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False
        self._called_stop = False

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @property
    def sim(self) -> FindViewSim:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_info(self) -> dict:
        return {
            "episode_id": self._current_episode.episode_id,
            "img_name": self._current_episode.img_name,
            "path": self._current_episode.path,
            "label": self._current_episode.label,
            "sub_label": self._current_episode.sub_label,
            "difficulty": self._current_episode.difficulty,
            "initial_rotation": self._current_episode.initial_rotation,
            "target_rotation": self._current_episode.target_rotation,
            "current_rotation": self._rot_tracker.rot,
            "steps_for_shortest_path": self._current_episode.steps_for_shortest_path,
            "elapsed_steps": self._elapsed_steps,
            "called_stop": self._called_stop,
        }

    def get_metrics(self):
        """NOTE: should return dict of metrics for reward calculation, etc...
        """

        # NOTE: get basic info (mainly from episode and env)
        info = self.get_info()

        # NOTE: this might be heavy...
        # Great for debugging:
        history = {
            "rotation_history": self._rot_tracker.history,
        }

        # NOTE: do some metric calculations
        same_rots = count_same_rots(self._rot_tracker.history)
        distances = distance_to_target(
            target_rotation=self._current_episode.target_rotation,
            current_rotation=self._rot_tracker.rot,
        )

        return {
            **info,
            **history,
            **distances,
            **same_rots,
        }

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
        self._called_stop = False

    def _make_base_obs(
        self,
        pers,
        target,
    ) -> dict:
        obs = {
            "pers": pers,
            "target": target,
        }
        return obs

    def reset(self) -> None:

        self._reset_stats()

        # FIXME: what to do when iterator is finished and `reset` is called?
        self._current_episode = next(self._episode_iterator)

        initial_rotation = self._current_episode.initial_rotation
        target_rotation = self._current_episode.target_rotation
        episode_path = self._current_episode.path

        self._rot_tracker.initialize(initial_rotation)
        pers, target = self._sim.reset(
            equi_path=episode_path,
            initial_rotation=initial_rotation,
            target_rotation=target_rotation,
        )

        return self._make_base_obs(
            pers=pers,
            target=target,
        )

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = self._called_stop
        if self._past_limit():
            self._episode_over = True

    def step_before(self, action: str):
        assert self._episode_start_time is not None, \
            "Cannot call step before calling reset"

        assert self._episode_over is False, \
            "ERR: Episode is over, call reset before calling step"

        # FIXME: usually I would check if action is in the list, but I'm lazy
        # FIXME: add support for integer and other action formats
        if action == FindViewActions.STOP:
            self._called_stop = True
            rot = None
        else:
            rot = self._rot_tracker.convert(action)

        return rot

    def step_after(self):
        pers = self._sim.pers
        target = self._sim.target

        obs = self._make_base_obs(
            pers=pers,
            target=target,
        )

        self._update_step_stats()

        return obs

    def step(self, action: Union[Dict[str, str], str]):

        assert self._episode_start_time is not None, \
            "Cannot call step before calling reset"

        assert self._episode_over is False, \
            "ERR: Episode is over, call reset before calling step"

        if isinstance(action, dict):
            action = action['action']

        # FIXME: usually I would check if action is in the list, but I'm lazy
        # FIXME: add support for integer and other action formats
        if action == FindViewActions.STOP:
            pers = self._sim.pers
            self._called_stop = True
        else:
            rot = self._rot_tracker.convert(action)
            pers = self._sim.move(rot=rot)

        target = self._sim.target

        obs = self._make_base_obs(
            pers=pers,
            target=target,
        )

        self._update_step_stats()

        return obs

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def render(self):
        pers = self._sim.render_pers()
        target = self._sim.render_target()
        return {
            "pers": pers,
            "target": target,
        }

    def change_difficulty(self, difficulty: str) -> None:
        # FIXME: need to implement for iterators
        assert self._sampler is not None, \
            "Sampler is None, maybe you're using an iterator?"
        self._sampler.set_difficulty(difficulty=difficulty)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

        self._min_steps = self._cfg.min_steps
        self._max_steps = self._cfg.max_steps
        self._rl_env_cfg = self._cfg.rl_env_cfgs
        self._slack_reward = self._rl_env_cfg['slack_reward']
        self._success_reward = self._rl_env_cfg['success_reward']

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
def make_env(
    cfg: Config,
    split: str,
    filter_fn=None,
    is_torch: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> FindViewEnv:
    if is_torch:
        assert dtype in (torch.float16, torch.float32, torch.float64)
    else:
        assert dtype in (np.float32, np.float64)
    env = FindViewEnv(
        cfg=cfg,
        split=split,
        filter_fn=filter_fn,
        is_torch=is_torch,
        dtype=dtype,
        device=device,
    )
    return env


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
