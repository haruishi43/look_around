#!/usr/bin/env python3

from copy import deepcopy
import os
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from LookAround.FindView.actions import FindViewActions
from LookAround.utils.visualizations import images_to_video, images_to_video_cv2

ASSET_ROOT = './LookAround/FindView/utils/assets'
ASSETS = {
    'up': os.path.join(ASSET_ROOT, 'up.png'),
    'down': os.path.join(ASSET_ROOT, 'down.png'),
    'right': os.path.join(ASSET_ROOT, 'right.png'),
    'left': os.path.join(ASSET_ROOT, 'left.png'),
    'stop': os.path.join(ASSET_ROOT, 'stop.png'),
}


def resize_and_add_bbox(
    img: np.ndarray,
    resize_shape: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Create a boundary around the input image
    """
    template = np.full(
        (
            resize_shape[0] + 2 * thickness,
            resize_shape[1] + 2 * thickness,
            3,
        ),
        color,
        dtype=np.uint8,
    )
    if img.shape[:2] != resize_shape:
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)

    h, w, _ = img.shape
    template[
        thickness : thickness + h,
        thickness : thickness + w,
    ] = img
    return template


def draw_bfov(
    equi: np.ndarray,
    points: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray:
    """Draw a bounding FOV around the prespective view
    """

    points = points.tolist()
    points = [(x, y) for y, x in points]
    for index, point in enumerate(points):
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]

        if abs(point[0] - next_point[0]) < 100 and abs(point[1] - next_point[1]) < 100:
            # NOTE: make sure that lines are cut when the BFOV wraps around
            cv2.line(equi, point, next_point, color=color, thickness=thickness)

    return equi


def generate_movement_video(
    output_dir: str,
    video_name: str,
    equi: np.ndarray,
    pers: List[np.ndarray],
    target: np.ndarray,
    pers_bboxs: List[np.ndarray],
    target_bbox: np.ndarray,
    actions: List[Union[int, str, Dict[str, int]]],
    equi_size: Tuple[int, int] = (512, 1024),
    pers_size: Tuple[int, int] = (256, 256),
    pers_boundary_thickness: int = 5,
    init_color: Tuple[int, int, int] = (218, 62, 82),
    pers_color: Tuple[int, int, int] = (92, 187, 255),
    target_color: Tuple[int, int, int] = (150, 230, 179),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    frame_boundary: Tuple[int, int] = (6, 8),
    boundary_between_updown: int = 10,
    boundary_between_pers: int = 20,
    boundary_arrow: int = 10,
    use_imageio: bool = False,
    fps: int = 30,
    quality: int = 5,
    **kwargs,
) -> None:
    """Generate an "easy to see" movement video
    """

    assert len(pers) == len(pers_bboxs)
    assert len(pers) == len(actions)

    # 1. create base template to place the images
    frames = []
    template_frame = np.full(
        (
            2 * frame_boundary[0] + equi_size[0]
            + boundary_between_updown + pers_size[0]
            + 2 * pers_boundary_thickness,
            2 * frame_boundary[1] + equi_size[1],
            3,
        ),
        background_color,
        dtype=np.uint8,
    )
    if equi.shape[:2] != equi_size:
        equi = cv2.resize(equi, equi_size, interpolation=cv2.INTER_LINEAR)

    # NOTE: make sure that the video is a multiple of 16 for imageio

    # 2. resize and put boundary on perspective images
    initial_pers = deepcopy(pers[0])
    initial_bpers = resize_and_add_bbox(
        img=initial_pers,
        resize_shape=pers_size,
        color=init_color,
        thickness=pers_boundary_thickness,
    )
    bpers = []
    for p in pers:
        bpers.append(
            resize_and_add_bbox(
                img=p,
                resize_shape=pers_size,
                color=pers_color,
                thickness=pers_boundary_thickness,
            )
        )
    target_bpers = resize_and_add_bbox(
        img=target,
        resize_shape=pers_size,
        color=target_color,
        thickness=pers_boundary_thickness,
    )
    pers_size = (
        pers_size[0] + 2 * pers_boundary_thickness,
        pers_size[1] + 2 * pers_boundary_thickness,
    )

    # 3. load action images
    # assume that we have smaller space width-wise
    left_over_w = template_frame.shape[1] - (
        2 * frame_boundary[1]
        + 2 * boundary_between_pers
        + 3 * pers_size[1]
        + 2 * boundary_arrow
    )
    assert left_over_w > 128, "space too low for directions"
    if left_over_w > pers_size[0]:
        # if the left over width is larger than pers height
        side = pers_size[0]
        displacement = (0, (left_over_w - pers_size[1]) // 2)
    else:
        side = left_over_w
        displacement = ((pers_size[0] - left_over_w) // 2, 0)
    asset_size = (side, side)
    assets = {}
    for name, asset_path in ASSETS.items():
        asset = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        asset = cv2.resize(asset, asset_size, interpolation=cv2.INTER_NEAREST)
        background = np.full(
            (*asset.shape[:2], 3),
            background_color,
            dtype=np.uint8,
        )
        alpha = cv2.split(asset)[3][..., None] // 255  # assume 0 or 1
        asset = cv2.cvtColor(asset, cv2.COLOR_BGRA2BGR)
        foreground = alpha * asset
        background = (1 - alpha) * background
        asset = foreground + background
        assets[name] = asset

    # 4. draw initial and target BFOV
    initial = deepcopy(pers_bboxs[0])
    base_equi = draw_bfov(
        equi=deepcopy(equi),
        points=initial,
        color=init_color,
        thickness=4,
    )
    base_equi = draw_bfov(
        equi=base_equi,
        points=target_bbox,
        color=target_color,
        thickness=4,
    )

    # 5. draw BFOV for each perspectives and add it to the template
    for index, (bp, bbox) in tqdm(enumerate(zip(bpers, pers_bboxs))):
        _equi = draw_bfov(
            equi=deepcopy(base_equi),
            points=bbox,
            color=pers_color,
            thickness=2,
        )

        frame = deepcopy(template_frame)
        # equirectangular image
        frame[
            frame_boundary[0] : frame_boundary[0] + equi_size[0],
            frame_boundary[1] : frame_boundary[1] + equi_size[1],
        ] = _equi

        h = frame_boundary[0] + equi_size[0] + boundary_between_updown
        w = frame_boundary[1]
        # initial perspective
        frame[
            h : h + pers_size[0],
            w : w + pers_size[1],
        ] = initial_bpers
        w += pers_size[1] + boundary_between_pers
        # target perspective
        frame[
            h : h + pers_size[0],
            w : w + pers_size[1],
        ] = target_bpers
        w += pers_size[1] + boundary_between_pers
        # current perspective
        frame[
            h : h + pers_size[0],
            w : w + pers_size[1],
        ] = bp
        w += pers_size[1] + boundary_arrow

        # process actions
        action = actions[index]
        if isinstance(action, dict):
            action = action['action']
        if isinstance(action, int):
            action = FindViewActions.all[action]
        asset = assets[action]
        h += displacement[0]
        w += displacement[1]
        frame[
            h: h + asset_size[0],
            w: w + asset_size[1],
        ] = asset

        if use_imageio:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    if use_imageio:
        images_to_video(
            frames,
            output_dir=output_dir,
            video_name=video_name,
            fps=fps,
            quality=quality,
            verbose=False,
            **kwargs,
        )
    else:
        images_to_video_cv2(
            images=frames,
            output_dir=output_dir,
            video_name=video_name,
            fps=fps,
            **kwargs,
        )
