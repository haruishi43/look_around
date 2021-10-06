#!/usr/bin/env python3

from typing import Optional, Tuple

from LookAround.FindView.vec_env import VecEnv


class DifficultyScheduler(object):

    # Properties
    current_level: int = 0
    difficulties: Tuple[str] = ('easy', 'medium', 'hard')  # NOTE: notice the order

    def __init__(
        self,
        initial_difficulty: str,
        update_interval: int,
        num_updates_done: int = 0,
        bounded: bool = False,
        difficulties: Optional[Tuple[str]] = None,
    ) -> None:

        if difficulties is not None:
            assert isinstance(difficulties, tuple) and len(difficulties) > 0
            self.difficulties = difficulties

        self.update_interval = update_interval
        assert initial_difficulty in self.difficulties
        self.current_level = self.difficulties.index(initial_difficulty)
        self._update_difficulty(num_updates_done=num_updates_done)
        self.bounded = bounded

    @property
    def current_difficulty(self) -> str:
        return self.difficulties[self.current_level]

    def _update_difficulty(self, num_updates_done: int) -> bool:
        _level = num_updates_done // self.update_interval
        if _level > self.current_level:
            if _level > len(self.difficulties) - 1:
                self.current_level = len(self.difficulties) - 1
            else:
                self.current_level = _level

            return True
        else:
            return False

    def update(self, envs: VecEnv, num_updates_done: int) -> None:
        # FIXME: add update by metrics
        # need to save metrics
        # - doesn't work well when resume is turned on
        # - what threshold should we use and which metric?
        if self._update_difficulty(num_updates_done=num_updates_done):
            envs.change_difficulty(
                difficulty=self.difficulties[self.current_level],
                bounded=self.bounded,
            )
