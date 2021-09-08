#!/usr/bin/env python3

from functools import lru_cache, partial
from typing import Any, Dict, Tuple

import numpy as np

from LookAround.FindView.dataset.episode import Episode, PseudoEpisode


def normal_distribution(
    normalized_arr: np.ndarray,
    mu: float = 0.0,
    sigma: float = 0.3,
) -> np.ndarray:
    probs = (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((normalized_arr - mu) ** 2) / (2 * sigma ** 2))
    )
    probs = probs / probs.sum()
    return probs


@lru_cache(maxsize=128)
def get_pitch_range(
    threshold: int,
    mu: float = 0.0,
    sigma: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    phis = np.arange(-threshold, threshold + 1)
    prob_phis = normal_distribution(
        phis / threshold,
        mu=mu,
        sigma=sigma,
    )
    return phis, prob_phis


@lru_cache(maxsize=128)
def get_yaw_range() -> np.ndarray:
    thetas = np.arange(-180 + 1, 180 + 1)
    return thetas


def find_minimum(diff_yaw):
    """Because yaw wraps around, we have to take the minimum distance
    """
    if diff_yaw > 180:
        diff_yaw = 360 - diff_yaw
    return diff_yaw


def l1_dist(abs_x, abs_y):
    # grid distance -> how many steps
    return abs_x + abs_y


def l2_dist(abs_x, abs_y):
    return np.sqrt(abs_x**2 + abs_y**2)


def base_condition(
    init_pitch,
    init_yaw,
    targ_pitch,
    targ_yaw,
    min_steps,
    max_steps,
    step_size,
):
    diff_pitch = np.abs(init_pitch - targ_pitch)
    diff_yaw = find_minimum(np.abs(init_yaw - targ_yaw))
    l1 = l1_dist(diff_pitch, diff_yaw)
    return (
        int(l1) % step_size == 0
        and l1 > min_steps * step_size
        and l1 < max_steps * step_size
    )


def easy_condition(
    init_pitch,
    init_yaw,
    targ_pitch,
    targ_yaw,
    fov,
):
    # of course, this isn't accurate, but we just assume height is less than width
    max_l2 = l2_dist(fov / 2, fov / 2)

    diff_pitch = np.abs(init_pitch - targ_pitch)
    diff_yaw = find_minimum(np.abs(init_yaw - targ_yaw))

    l2 = l2_dist(diff_pitch, diff_yaw)
    return l2 <= max_l2


def medium_condition(
    init_pitch,
    init_yaw,
    targ_pitch,
    targ_yaw,
    fov,
):
    diff_pitch = np.abs(init_pitch - targ_pitch)
    diff_yaw = find_minimum(np.abs(init_yaw - targ_yaw))
    return (
        diff_yaw > fov / 2
        and diff_yaw <= fov
        and diff_pitch <= fov
    )


def hard_condition(
    init_pitch,
    init_yaw,
    targ_pitch,
    targ_yaw,
    fov,
):
    diff_pitch = np.abs(init_pitch - targ_pitch)
    diff_yaw = find_minimum(np.abs(init_yaw - targ_yaw))

    return (
        diff_yaw > fov
        or diff_pitch > fov
    )


class Sampler(object):

    def __call__(self, pseudo) -> Episode:
        raise NotImplementedError


class DifficultySampler(Sampler):

    difficulties = ('easy', 'medium', 'hard')

    def __init__(
        self,
        difficulty: str,
        fov: float,
        min_steps: int,
        max_steps: int,
        step_size: int,
        threshold: int,
        seed: int,
        mu: float = 0.0,
        sigma: float = 0.3,
        num_tries: int = 100000,
    ) -> None:

        self.set_difficulty(difficulty)
        self.fov = fov
        self.min_steps = min_steps
        self.max_steps = max_steps,
        self.step_size = step_size
        self.threshold = threshold
        self.num_tries = num_tries

        pitches, prob = get_pitch_range(threshold=threshold, mu=mu, sigma=sigma)
        yaws = get_yaw_range()
        self.pitches = pitches
        self.prob = prob
        self.yaws = yaws

        self.base_cond = partial(
            base_condition,
            min_steps=min_steps,
            max_steps=max_steps,
            step_size=step_size,
        )
        self.prev_kwargs = None

        self.seed(seed)

    def __call__(self, pseudo: PseudoEpisode) -> Episode:

        kwargs = self.sample()
        self.prev_kwargs = kwargs

        episode = Episode(
            episode_id=0,  # placeholder
            img_name=pseudo.img_name,
            path=pseudo.path,
            label=pseudo.label,
            sub_label=pseudo.sub_label,
            **kwargs,
        )

        return episode

    def sample(self) -> Dict[str, Any]:
        difficulty = self.get_difficulty()

        if difficulty == "easy":
            cond = partial(easy_condition, fov=self.fov)
        elif difficulty == "medium":
            cond = partial(medium_condition, fov=self.fov)
        elif difficulty == "hard":
            cond = partial(hard_condition, fov=self.fov)
        else:
            raise ValueError(f"ERR: unknown difficulty {difficulty}")

        _count = 0  # FIXME: how to deal with criteria that's REALLY hard?
        while True:
            # sample rotations
            init_pitch = int(np.random.choice(self.pitches, p=self.prob))
            init_yaw = int(np.random.choice(self.yaws))

            targ_pitch = int(np.random.choice(self.pitches, p=self.prob))
            targ_yaw = int(np.random.choice(self.yaws))

            if (
                self.base_cond(init_pitch, init_yaw, targ_pitch, targ_yaw)
                and cond(init_pitch, init_yaw, targ_pitch, targ_yaw)
            ):
                kwargs = {
                    "initial_rotation": {
                        "roll": 0,
                        "pitch": init_pitch,
                        "yaw": init_yaw,
                    },
                    "target_rotation": {
                        "roll": 0,
                        "pitch": targ_pitch,
                        "yaw": targ_yaw,
                    },
                    "difficulty": difficulty,
                    "steps_for_shortest_path": int(
                        np.abs(init_pitch - targ_pitch) + np.abs(init_yaw - targ_yaw)
                    ),  # NOTE: includes `stop` action
                }
                break

            _count += 1
            if _count > self.num_tries:
                assert self.prev_kwargs is not None, \
                    f"ERR: criteria is hard from the beginning; couldn't smaple in {self.num_tries}"
                kwargs = self.prev_kwargs

        return kwargs

    def get_difficulty(self):
        if self.difficulty == 'medium':
            return np.random.choice(('easy', 'medium'))
        elif self.difficulty == 'hard':
            return np.random.choice(('easy', 'medium', 'hard'))
        else:
            return 'easy'

    def set_difficulty(self, difficulty: str):
        assert difficulty in self.difficulties
        self.difficulty = difficulty

    def seed(self, seed: int) -> None:
        np.random.seed(seed)