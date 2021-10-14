#!/usr/bin/env python3

from typing import Tuple

import numpy as np

from LookAround.FindView.sim import FindViewSim


def resize_and_add_bbox(
    img: np.ndarray,
    resize_shape: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    ...


def draw_bfov(
    equi,
    points: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray:

    points = points.tolist()
    points = [(x, y) for y, x in points]

    for index, point in enumerate(points):
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]

        if abs(point[0] - next_point[0]) < 100 and abs(point[1] - next_point[1]) < 100:
            cv2.line(equi, point, next_point, color=color, thickness=thickness)

    return equi


def draw_bfov_video(
    sim: FindViewSim,
    history: list,
    target: dict,
):

    frames = []
    initial = history.pop(0)
    equi = draw_bfov(
        equi=sim.render_equi(to_bgr=True),
        points=sim.get_bounding_fov(initial),
        color=(0, 255, 38),
        thickness=5,
    )
    equi = draw_bfov(
        equi=equi,
        points=sim.get_bounding_fov(target),
        color=(255, 0, 43),
        thickness=5,
        input_cv2=True,
    )
    frames.append(equi)

    for rot in history:
        frame = deepcopy(equi)

        frame = draw_bfov(
            equi=frame,
            points=sim.get_bounding_fov(rot),
            color=(0, 162, 255),
            thickness=3,
            input_cv2=True,
        )
        frames.append(frame)

    return frames
