#!/usr/bin/env python3

from functools import lru_cache
from typing import Tuple

import numpy as np


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
