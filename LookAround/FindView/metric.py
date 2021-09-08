#!/usr/bin/env python3

"""Calculate useful metrics

"""

from collections import Counter

import numpy as np


def count_same_rots(history):
    # convert history to [(pitch, yaw)]
    t = [(r['pitch'], r['yaw']) for r in history]
    occurrences = Counter(t)
    num_repeated = sum([v for v in occurrences.values() if v > 1])
    total = sum(occurrences.values())
    if total == 0:
        return {
            "num_same_view": 0.,
        }
    else:
        return {
            "num_same_view": num_repeated / total,
        }


def distance_to_target(
    target_rotation,
    current_rotation,
) -> dict:

    # FIXME: calculate something...
    l1_distance_to_target = l1_distance(current_rotation, target_rotation)
    l2_distance_to_target = l2_distance(current_rotation, target_rotation)

    return {
        "l1_distance_to_target": l1_distance_to_target,
        "l2_distance_to_target": l2_distance_to_target,
    }


def find_minimum(diff_yaw):
    """Because yaw wraps around, we have to take the minimum distance
    """
    if diff_yaw > 180:
        diff_yaw = 360 - diff_yaw
    return diff_yaw


def l1_distance(r1, r2):
    r1p = r1['pitch']
    r1y = r1['yaw']
    r2p = r2['pitch']
    r2y = r2['yaw']

    dp = np.abs(r1p - r2p)
    dy = find_minimum(np.abs(r1y - r2y))
    return int(dp + dy)


def l2_distance(r1, r2):
    r1p = r1['pitch']
    r1y = r1['yaw']
    r2p = r2['pitch']
    r2y = r2['yaw']

    dp = np.abs(r1p - r2p)
    dy = find_minimum(np.abs(r1y - r2y))
    return float(np.sqrt(dp**2 + dy**2))
