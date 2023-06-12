#!/usr/bin/env python3

"""Agent specific visualizations
"""

import cv2
import numpy as np


def draw_matches(
    gray_pers: np.ndarray,
    kps_pers,
    gray_target: np.ndarray,
    kps_target,
    matches,
) -> np.ndarray:
    img = cv2.drawMatches(
        img1=gray_pers,
        keypoints1=kps_pers,
        img2=gray_target,
        keypoints2=kps_target,
        matches1to2=matches,
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return img
