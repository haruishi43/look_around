#!/usr/bin/env python3

from functools import partial
from typing import Dict, List, Union

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

    equi: Tensor
    target: Tensor
    pers: Tensor
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
        self.equi = None
        self.target = None
        self.pers = None
        self.equi_path = None
        self.initial_rotation = None
        self.target_rotation = None

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
            self.equi = self._load_func(img_path=equi_path)

        # initialize data
        self.equi_path = equi_path
        self.initial_rotation = initial_rotation
        self.target_rotation = target_rotation

        # set images
        self.pers = self.sample(rot=initial_rotation)
        self.target = self.sample(rot=target_rotation)

    def sample(
        self,
        rot: Rots,
    ) -> Tensor:
        # NOTE: convert deg to rad
        rad_rot = deg2rad(rot)
        return self.equi2pers(copy_tensor(self.equi), rots=rad_rot)

    def move(self, rot: Rots) -> Tensor:
        """Rotate view and return unrefined view
        """
        self.pers = self.sample(rot=rot)
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

    def render_pers(self) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(copy_tensor(self.pers))
        else:
            return post_process_for_render(copy_tensor(self.pers))

    def render_target(self) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(copy_tensor(self.target))
        else:
            return post_process_for_render(copy_tensor(self.target))

    def __del__(self):
        # NOTE: clean up
        del self.equi
        del self.target
        del self.pers
        del self.equi2pers

    def close(self) -> None:
        pass


def batch_sample(sims: List[FindViewSim], rots: List[Rots]) -> Tensor:
    rad_rots = [deg2rad(rot) for rot in rots]

    if sims[0].is_torch:
        batched_equi = torch.stack([s.equi.clone() for s in sims], dim=0)
    else:
        batched_equi = np.stack([s.equi.copy() for s in sims], axis=0)

    batched_pers = sims[0].equi2pers(batched_equi, rots=rad_rots)

    for i, sim in enumerate(sims):
        sim.pers = batched_pers[i]

    return batched_pers
