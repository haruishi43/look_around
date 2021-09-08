#!/usr/bin/env python3

import random
import time

from LookAround.FindView.dataset.episode import Episode
from typing import Dict, Optional, Union

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
from LookAround.FindView.actions import FindViewActions
from LookAround.FindView.rotation_tracker import RotationTracker
from LookAround.FindView.dataset import make_dataset
from LookAround.FindView.dataset.static_dataset import StaticDataset, StaticIterator
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset, DynamicGenerator
from LookAround.FindView.dataset.sampling import DifficultySampler, Sampler


class FindViewEnv(object):

    number_of_episodes: Optional[int]

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
            self.number_of_episodes = None
        else:
            self._sampler = None
            iter_options = self._cfg.episode_iterator_kwargs
            iter_options['seed'] = self._cfg.seed
            self._episode_iterator = self._dataset.get_iterator(
                **iter_options,
            )
            self.number_of_episodes = len(self._dataset.episodes)

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
                            3,
                            self._cfg.sim.height,
                            self._cfg.sim.width,
                        ),
                    ),
                    "target": spaces.Box(
                        low=torch.finfo(dtype).min,
                        high=torch.finfo(dtype).max,
                        shape=(
                            3,
                            self._cfg.sim.height,
                            self._cfg.sim.width,
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
                            3,
                            self._cfg.sim.height,
                            self._cfg.sim.width,
                        ),
                    ),
                    "target": spaces.Box(
                        low=np.finfo(dtype).min,
                        high=np.finfo(dtype).max,
                        shape=(
                            3,
                            self._cfg.sim.height,
                            self._cfg.sim.width,
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
        # history = {
        #     "rotation_history": self._rot_tracker.history,
        # }

        # NOTE: do some metric calculations
        same_rots = count_same_rots(self._rot_tracker.history)
        distances = distance_to_target(
            target_rotation=self._current_episode.target_rotation,
            current_rotation=self._rot_tracker.rot,
        )

        return {
            **info,
            # **history,
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

        if isinstance(action, int):
            action = FindViewActions.all[action]

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
