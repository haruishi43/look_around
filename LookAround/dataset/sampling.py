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
