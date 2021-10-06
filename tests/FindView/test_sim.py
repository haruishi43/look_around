#!/usr/bin/env python3

"""
Test `rotation_tracker` and `sim`

"""

import os
from typing import Union

import cv2
import numpy as np
import pytest
import torch

from LookAround.FindView.rotation_tracker import RotationTracker
from LookAround.FindView.sim import FindViewSim

DATA_ROOT = "tests/data/sun360"


@pytest.mark.parametrize('inc', [1, 2, 5])
@pytest.mark.parametrize('threshold', [60, 90])
def test_rotation_tracker(
    inc: int,
    threshold: int,
) -> None:

    initial_rotation = dict(
        roll=0,
        pitch=0,
        yaw=0,
    )

    rot_tracker = RotationTracker(
        inc=inc,
        pitch_threshold=threshold,
    )

    # move up until it hits the threshold
    rot_tracker.reset(initial_rotation=initial_rotation)
    ex_pitch = initial_rotation['pitch']  # should be a copy right?
    while True:
        rot = rot_tracker.move('up')
        pitch = rot['pitch']
        if pitch == threshold:
            break

        ex_pitch += inc
        assert ex_pitch != threshold, \
            f"pitch ({pitch}) should have reached threshold already"

    # move down until it hits the threshold
    rot_tracker.reset(initial_rotation=initial_rotation)
    ex_pitch = initial_rotation['pitch']
    while True:
        rot = rot_tracker.move('down')
        pitch = rot['pitch']
        if pitch == -threshold:
            break

        ex_pitch -= inc
        assert ex_pitch != -threshold, \
            f"pitch ({pitch}) should have reached threshold already"

    # move right, it should wrap around to the original position
    rot_tracker.reset(initial_rotation=initial_rotation)
    ex_yaw = initial_rotation['yaw']
    while True:
        rot = rot_tracker.move('right')
        yaw = rot['yaw']
        if yaw == initial_rotation['yaw']:
            break

        ex_yaw += inc
        if ex_yaw > 180:
            ex_yaw -= 2 * 180
        assert ex_yaw != initial_rotation['yaw'], \
            f"yaw ({yaw}) should have went full circles already"

    # move left, it should wrap around to the original position
    rot_tracker.reset(initial_rotation=initial_rotation)
    ex_yaw = initial_rotation['yaw']
    while True:
        rot = rot_tracker.move('left')
        yaw = rot['yaw']
        if yaw == initial_rotation['yaw']:
            break

        ex_yaw -= inc
        if ex_yaw <= -180:
            ex_yaw += 2 * 180
        assert ex_yaw != initial_rotation['yaw'], \
            f"yaw ({yaw}) should have went full circles already"


@pytest.mark.parametrize('height', [128])
@pytest.mark.parametrize('width', [128, 256])
@pytest.mark.parametrize('fov', [90.0])
@pytest.mark.parametrize('dtype', [torch.float32, np.float32])
def test_cpu_sim(
    height: int,
    width: int,
    fov: float,
    dtype: Union[np.dtype, torch.dtype],
) -> None:

    cv_write = False

    if dtype in (np.float32, np.float64):
        is_torch = False
    elif dtype in (torch.float32, torch.float64):
        is_torch = True

    # initialize module and loader
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode="bilinear",
    )
    sim.inititialize_loader(dtype=dtype)

    equi_path = os.path.join(
        DATA_ROOT,
        "indoor",
        "bedroom",
        "pano_aaacisrhqnnvoq.jpg",
    )
    initial_rotation = dict(
        roll=0,
        pitch=0,
        yaw=0,
    )
    target_rotation = dict(
        roll=0,
        pitch=0,
        yaw=30,
    )

    # reset the simulator
    pers, target = sim.reset(
        equi_path=equi_path,
        initial_rotation=initial_rotation,
        target_rotation=target_rotation,
    )

    assert torch.is_tensor(pers) == is_torch
    assert isinstance(pers, np.ndarray) == (not is_torch)
    assert torch.is_tensor(target) == is_torch
    assert isinstance(target, np.ndarray) == (not is_torch)
    assert pers.shape == (3, height, width)
    assert target.shape == (3, height, width)
    assert sim.height == height
    assert sim.width == width
    assert sim.fov == fov

    equi_path = os.path.join(
        DATA_ROOT,
        "indoor",
        "bedroom",
        "pano_aaawcsqcbquzht.jpg",
    )
    initial_rotation = dict(
        roll=0,
        pitch=10,
        yaw=20,
    )
    target_rotation = dict(
        roll=0,
        pitch=-10,
        yaw=-30,
    )

    # reset the simulator
    pers, target = sim.reset(
        equi_path=equi_path,
        initial_rotation=initial_rotation,
        target_rotation=target_rotation,
    )

    assert torch.is_tensor(pers) == is_torch
    assert isinstance(pers, np.ndarray) == (not is_torch)
    assert torch.is_tensor(target) == is_torch
    assert isinstance(target, np.ndarray) == (not is_torch)
    assert pers.shape == (3, height, width)
    assert target.shape == (3, height, width)
    assert sim.height == height
    assert sim.width == width
    assert sim.fov == fov

    cv_pers = sim.render_pers()
    cv_target = sim.render_target()

    assert cv_pers.shape == (height, width, 3)
    assert cv_target.shape == (height, width, 3)

    pers = sim.move(dict(roll=0, pitch=10, yaw=0))

    cv_pers2 = sim.render_pers()

    if cv_write:
        # save visuals for debugging
        save_root = "tests/results"
        cv2.imwrite(os.path.join(save_root, "pers1.jpg"), cv_pers)
        cv2.imwrite(os.path.join(save_root, "target.jpg"), cv_target)
        cv2.imwrite(os.path.join(save_root, "pers2.jpg"), cv_pers2)


@pytest.mark.parametrize('height', [128])
@pytest.mark.parametrize('width', [128, 256])
@pytest.mark.parametrize('fov', [90.0])
def test_gpu_sim(
    height: int,
    width: int,
    fov: float,
) -> None:

    cv_write = False
    dtype = torch.float32
    device = torch.device(0)

    # initialize module and loader
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode="bilinear",
    )
    sim.inititialize_loader(
        dtype=dtype,
        device=device,
    )

    equi_path = os.path.join(
        DATA_ROOT,
        "indoor",
        "bedroom",
        "pano_aaacisrhqnnvoq.jpg",
    )
    initial_rotation = dict(
        roll=0,
        pitch=0,
        yaw=0,
    )
    target_rotation = dict(
        roll=0,
        pitch=0,
        yaw=30,
    )

    # reset the simulator
    pers, target = sim.reset(
        equi_path=equi_path,
        initial_rotation=initial_rotation,
        target_rotation=target_rotation,
    )

    assert torch.is_tensor(pers)
    assert torch.is_tensor(target)
    assert pers.shape == (3, height, width)
    assert target.shape == (3, height, width)
    assert sim.height == height
    assert sim.width == width
    assert sim.fov == fov

    equi_path = os.path.join(
        DATA_ROOT,
        "indoor",
        "bedroom",
        "pano_aaawcsqcbquzht.jpg",
    )
    initial_rotation = dict(
        roll=0,
        pitch=10,
        yaw=20,
    )
    target_rotation = dict(
        roll=0,
        pitch=-10,
        yaw=-30,
    )

    # reset the simulator
    pers, target = sim.reset(
        equi_path=equi_path,
        initial_rotation=initial_rotation,
        target_rotation=target_rotation,
    )

    assert torch.is_tensor(pers)
    assert torch.is_tensor(target)
    assert pers.shape == (3, height, width)
    assert target.shape == (3, height, width)
    assert sim.height == height
    assert sim.width == width
    assert sim.fov == fov

    cv_pers = sim.render_pers()
    cv_target = sim.render_target()

    assert cv_pers.shape == (height, width, 3)
    assert cv_target.shape == (height, width, 3)

    pers = sim.move(dict(roll=0, pitch=10, yaw=0))

    cv_pers2 = sim.render_pers()

    if cv_write:
        # save visuals for debugging
        save_root = "tests/results"
        cv2.imwrite(os.path.join(save_root, "torch_pers1.jpg"), cv_pers)
        cv2.imwrite(os.path.join(save_root, "torch_target.jpg"), cv_target)
        cv2.imwrite(os.path.join(save_root, "torch_pers2.jpg"), cv_pers2)
