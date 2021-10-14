#!/usr/bin/env python3

import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

from gym import spaces
import numpy as np
import torch

from LookAround.FindView.metric import (
    count_same_rots,
    distance_to_target,
    path_efficiency,
)

from LookAround.config import Config
from LookAround.core.spaces import ActionSpace, EmptySpace
from LookAround.FindView.sim import FindViewSim, Tensor
from LookAround.FindView.actions import FindViewActions
from LookAround.FindView.rotation_tracker import RotationTracker
from LookAround.FindView.dataset import Episode, make_dataset
from LookAround.FindView.dataset.static_dataset import StaticDataset, StaticIterator
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset, DynamicGenerator


class FindViewEnv(object):

    # Properties
    action_space: ActionSpace
    observation_space: spaces.Dict

    # Modules
    _dataset: Union[DynamicDataset, StaticDataset]
    _episode_iterator: Union[DynamicGenerator, StaticIterator]
    _rot_tracker: RotationTracker
    _sim: FindViewSim

    # Hidden Properties
    _current_episode: Optional[Episode]
    _number_of_episodes: Optional[int]
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self,
        dataset: Union[DynamicDataset, StaticDataset],
        episode_iterator: Union[DynamicGenerator, StaticIterator],
        sim: FindViewSim,
        seed: Optional[int] = None,
    ) -> None:
        """FindView Environment
        """

        assert sim.fov == dataset.fov

        # FIXME: originally, I only wanted `dataset` and `sim` as the input,
        # but `episode_iterator` had to be there too since `episode_iterator`
        # can't be initialized without arguments
        self._dataset = dataset
        self._episode_iterator = episode_iterator
        self._sim = sim

        if isinstance(self._dataset, StaticDataset):
            self._number_of_episodes = len(self._dataset.episodes)
        elif isinstance(self._dataset, DynamicDataset):
            self._number_of_episodes = None
        else:
            raise ValueError("input dataset is not supported")

        # initialize rotation tracker
        self._rot_tracker = RotationTracker(
            inc=self._dataset.step_size,
            pitch_threshold=self._dataset.pitch_threshold,
        )

        # gym spaces
        self.action_space = ActionSpace(
            {
                action_name: EmptySpace()
                for action_name in FindViewActions.all
            }
        )
        dtype = self._sim.dtype
        if self._sim.is_torch:
            # in habitat they are known as `sensors`
            self.observation_space = spaces.Dict(
                {
                    "pers": spaces.Box(
                        low=torch.finfo(dtype).min,
                        high=torch.finfo(dtype).max,
                        shape=(
                            3,
                            self._sim.height,
                            self._sim.width,
                        ),
                    ),
                    "target": spaces.Box(
                        low=torch.finfo(dtype).min,
                        high=torch.finfo(dtype).max,
                        shape=(
                            3,
                            self._sim.height,
                            self._sim.width,
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
                            self._sim.height,
                            self._sim.width,
                        ),
                    ),
                    "target": spaces.Box(
                        low=np.finfo(dtype).min,
                        high=np.finfo(dtype).max,
                        shape=(
                            3,
                            self._sim.height,
                            self._sim.width,
                        ),
                    ),
                }
            )

        # initialize other variables
        self._max_episode_seconds = self._dataset.max_seconds
        self._max_episode_steps = self._dataset.max_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False
        self._called_stop = False

        if seed is not None and isinstance(seed, int):
            self.seed(seed)

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

        return cls(
            dataset=dataset,
            episode_iterator=episode_iterator,
            sim=sim,
            seed=cfg.seed,
        )

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @property
    def episodes(self) -> List[Episode]:
        assert isinstance(self._dataset, StaticDataset)
        return self._dataset.episodes

    @property
    def number_of_episodes(self) -> Optional[int]:
        return self._number_of_episodes

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
        # FIXME: do we really need this much info?
        return {
            "episode_id": self._current_episode.episode_id,
            "img_name": self._current_episode.img_name,
            "path": self._current_episode.path,
            "label": self._current_episode.label,
            "sub_label": self._current_episode.sub_label,
            "difficulty": self._current_episode.difficulty,
            "initial_rotation": self._current_episode.initial_rotation,
            "target_rotation": self._current_episode.target_rotation,
            "current_rotation": self._rot_tracker.current_rotation,
            "steps_for_shortest_path": self._current_episode.steps_for_shortest_path,
            "elapsed_steps": self._elapsed_steps,
            "called_stop": self._called_stop,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """should return dict of metrics for reward calculation, etc...
        """

        # NOTE: Only put the calculations needed by Benchmark and RLEnv
        # for RLEnv, do their own calculation by extending

        # NOTE: get basic info (mainly from episode and env)
        info = self.get_info()

        # NOTE: this might be heavy...
        # Great for debugging:
        # history = {
        #     "rotation_history": self._rot_tracker.history,
        # }

        # number of times the agent looked in the same direction (normalized)
        same_rots = count_same_rots(self._rot_tracker.history)
        same_rots_dict = dict(
            num_same_view=same_rots,
        )

        # Distances to target
        distances = distance_to_target(
            target_rotation=self._current_episode.target_rotation,
            current_rotation=self._rot_tracker.current_rotation,
        )
        distances_dict = dict(
            l1_distance=distances['l1_distance'],
            l2_distance=distances['l2_distance'],
        )

        # FIXME: might need to edit this metrics...
        # if efficiency is close to zero, it means that the agent is stopping
        # around the point of `steps_for_shortest_path` (doesn't mean the stopped place is close)
        # if efficiency is large, it means that the agent is looking around too much or struggling
        efficiency_dict = dict(
            efficiency=path_efficiency(
                shortest_path=info['steps_for_shortest_path'],
                steps=info['elapsed_steps'],
            ),
        )

        return {
            **info,
            # **history,
            **distances_dict,
            **same_rots_dict,
            **efficiency_dict,
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
        pers: Tensor,
        target: Tensor,
    ) -> Dict[str, Tensor]:
        obs = {
            "pers": pers,
            "target": target,
        }
        return obs

    def reset(self) -> Dict[str, Tensor]:

        self._reset_stats()

        self._current_episode = next(self._episode_iterator)
        assert self._current_episode is not None, "ERR: called reset, but there are no more episodes in the iterator"

        initial_rotation = self._current_episode.initial_rotation
        target_rotation = self._current_episode.target_rotation
        episode_path = self._current_episode.path

        self._rot_tracker.reset(initial_rotation)
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

        # unwrap actions
        if isinstance(action, dict):
            action = action['action']

        if isinstance(action, int):
            action = FindViewActions.all[action]

        if action == FindViewActions.STOP:
            self._called_stop = True
            rot = None
        else:
            rot = self._rot_tracker.move(action)

        return rot

    def step_after(self) -> Dict[str, Tensor]:
        pers = self._sim.pers
        target = self._sim.target

        obs = self._make_base_obs(
            pers=pers,
            target=target,
        )

        self._update_step_stats()

        return obs

    def step(self, action: Union[Dict[str, str], str]) -> Dict[str, Tensor]:

        assert self._episode_start_time is not None, \
            "Cannot call step before calling reset"

        assert self._episode_over is False, \
            "ERR: Episode is over, call reset before calling step"

        # unwrap actions
        if isinstance(action, dict):
            action = action['action']

        if isinstance(action, int):
            action = FindViewActions.all[action]

        if action == FindViewActions.STOP:
            pers = self._sim.pers
            self._called_stop = True
        else:
            rot = self._rot_tracker.move(action)
            pers = self._sim.move(rot=rot)

        target = self._sim.target

        obs = self._make_base_obs(
            pers=pers,
            target=target,
        )

        self._update_step_stats()

        return obs

    def seed(self, seed: int) -> None:
        # multi-thread safe
        self.rst = random.Random(seed)
        self.np_rst = np.random.RandomState(seed)

    def render(self, to_bgr: bool = True) -> Dict[str, np.ndarray]:
        pers = self._sim.render_pers(to_bgr=to_bgr)
        target = self._sim.render_target(to_bgr=to_bgr)
        return {
            "pers": pers,
            "target": target,
        }

    def change_difficulty(self, difficulty: str, bounded: bool) -> None:
        assert isinstance(self._dataset, DynamicDataset) \
            and isinstance(self._episode_iterator, DynamicGenerator)
        self._episode_iterator.set_difficulty(difficulty=difficulty, bounded=bounded)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
