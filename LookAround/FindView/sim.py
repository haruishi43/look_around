#!/usr/bin/env python3

from functools import partial
from typing import Dict, List, Optional, Union

from equilib import Equi2Pers

import numpy as np

import torch

from LookAround.core.improc import (
    load2numpy,
    load2torch,
    post_process_for_render,
    post_process_for_render_torch,
)

Tensor = Union[np.ndarray, torch.Tensor]
Rots = Dict[str, int]
Dtypes = Union[np.dtype, torch.dtype]


def copy_tensor(t: Tensor) -> Tensor:
    if isinstance(t, np.ndarray):
        return t.copy()
    elif torch.is_tensor(t):
        return t.clone()
    else:
        raise ValueError("ERR: cannot copy tensor")


def deg2rad(rot):
    return {
        "roll": 0.,
        "pitch": rot['pitch'] * np.pi / 180,
        "yaw": rot['yaw'] * np.pi / 180,
    }


class FindViewSim(object):

    _equi: Tensor
    _target: Tensor
    _pers: Tensor
    equi_path: str
    initial_rotation: Rots
    target_rotation: Rots

    _load_func = None

    def __init__(
        self,
        height: int,
        width: int,
        fov: float,
        sampling_mode: str,
    ) -> None:

        self.equi2pers = Equi2Pers(
            height=height,
            width=width,
            fov_x=fov,
            skew=0.0,
            z_down=True,
            mode=sampling_mode,
        )

        # initialize important variables to None
        self._equi = None
        self._target = None
        self._pers = None
        self.equi_path = None
        self.initial_rotation = None
        self.target_rotation = None

    @property
    def equi(self) -> Tensor:
        """Make sure that access to equi is always a clone"""
        return copy_tensor(self._equi)

    @property
    def target(self) -> Tensor:
        """Make sure that access to target is always a clone"""
        return copy_tensor(self._target)

    @property
    def pers(self) -> Tensor:
        """Make sure that access to pers is always a clone"""
        return copy_tensor(self._pers)

    def inititialize_loader(
        self,
        is_torch: bool,
        dtype: Dtypes,
        device: torch.device = torch.device('cpu'),
    ) -> None:

        # FIXME: how to make it so that it loads to some cuda device?

        self.is_torch = is_torch
        if is_torch:
            print(f"NOTE: Using loading to {device.type} with index: {device.index}")
            self._load_func = partial(
                load2torch,
                dtype=dtype,
                device=device,
                is_cv2=False,
            )
        else:
            self._load_func = partial(
                load2numpy,
                dtype=dtype,
                is_cv2=False
            )

    def initialize_from_episode(
        self,
        equi_path: str,
        initial_rotation: Rots,
        target_rotation: Rots,
    ) -> None:
        assert self._load_func is not None, \
            "ERR: loading function is not initialized"

        if equi_path != self.equi_path:
            # NOTE: only load equi when the equi_path differs
            self._equi = self._load_func(img_path=equi_path)

        # initialize data
        self.equi_path = equi_path
        self.initial_rotation = initial_rotation
        self.target_rotation = target_rotation

        # set images
        self._pers = self.sample(rot=initial_rotation)
        self._target = self.sample(rot=target_rotation)

    def sample(
        self,
        rot: Rots,
    ) -> Tensor:
        # NOTE: convert deg to rad
        rad_rot = deg2rad(rot)
        return self.equi2pers(self.equi, rots=rad_rot)

    def move(self, rot: Rots) -> Tensor:
        """Rotate view and return unrefined view
        """
        self._pers = self.sample(rot=rot)
        return self.pers

    def reset(
        self,
        equi_path: str,
        initial_rotation: Rots,
        target_rotation: Rots,
    ) -> Tensor:
        """Reset rotation and change equi image
        """

        # set the new episode data
        self.initialize_from_episode(
            equi_path=equi_path,
            initial_rotation=initial_rotation,
            target_rotation=target_rotation,
        )

        return self.pers, self.target

    def render_pers(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.pers, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.pers, to_bgr=to_bgr)

    def render_target(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.target, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.target, to_bgr=to_bgr)

    def render_equi(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.equi, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.equi, to_bgr=to_bgr)

    def __del__(self):
        # NOTE: clean up
        del self._equi
        del self._target
        del self._pers
        del self.equi2pers

    def close(self) -> None:
        pass


def batch_sample(sims: List[FindViewSim], rots: List[Optional[Rots]]):

    is_torch = sims[0].is_torch
    none_idx = [i for i, rot in enumerate(rots) if rot is None]

    rad_rots = [deg2rad(rot) for rot in rots if rot is not None]

    batched_equi = []
    for i, sim in enumerate(sims):
        if i not in none_idx:
            batched_equi.append(sim.equi)

    if len(batched_equi) == 0:
        return

    if is_torch:
        if len(batched_equi) == 1:
            batched_equi = batched_equi[0].unsqueeze(0)
        else:
            batched_equi = torch.stack(batched_equi, dim=0)
    else:
        if len(batched_equi) == 1:
            batched_equi = batched_equi[0][None, ...]
        else:
            batched_equi = np.stack(batched_equi, axis=0)

    assert len(batched_equi.shape) == 4
    batched_pers = sims[0].equi2pers(batched_equi, rots=rad_rots)
    assert len(batched_pers.shape) == 4

    count = 0
    for i, sim in enumerate(sims):
        # if `rot` was None, `sim` should keep the original `pers`
        if i not in none_idx:
            sim._pers = batched_pers[count]
            count += 1

    assert count == len(batched_pers)
