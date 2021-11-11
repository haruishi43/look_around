#!/usr/bin/env python3

from functools import partial
from typing import Dict, Optional, Union

from equilib import Equi2Pers
import numpy as np
import torch

from LookAround.config import Config
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


def deg2rad(rot: Rots):
    # NOTE: asserts that `pitch` and `yaw` are in rot
    return {
        "roll": 0.,
        "pitch": rot['pitch'] * np.pi / 180,
        "yaw": rot['yaw'] * np.pi / 180,
    }


class FindViewSim(object):

    is_torch: bool

    _equi: Tensor
    _target: Tensor
    _pers: Tensor
    _height: int
    _width: int
    _fov: float
    _dtype: Union[np.dtype, torch.dtype]
    _device: torch.device
    _equi_path: str
    _initial_rotation: Rots
    _target_rotation: Rots
    _load_func = None

    def __init__(
        self,
        height: int,
        width: int,
        fov: float,
        sampling_mode: str = "bilinear",
        skew: float = 0.0,
        z_down: bool = True,
    ) -> None:
        """FindView Simulator

        params:
        - height (int)
        - width (int)
        - fov (float): in degrees
        - sampling_mode (str): bilinear, chose from (bilinear, nearest, bicubic)
        - skew (float): 0.0
        - z_down (bool): changes the direction of pitch and yaw

        NOTE: intended usages
        - call `initialize_loader` to initialize image loader
        - call `reset` or `load_episode` with path and rotations
        - call `move` with rotation (returns perspective image as chw)
        - call `pers`, `target`, or `equi` to obtain current images as chw
        - call `render_*` to obtain cv2 format of the images

        NOTE: calling other methods externally could result in bugs
        """

        assert height <= width, \
            "unsupported image format, height must be equal or shorter than width"

        self.equi2pers = Equi2Pers(
            height=height,
            width=width,
            fov_x=fov,
            skew=skew,
            z_down=z_down,
            mode=sampling_mode,
        )

        self._height = height
        self._width = width
        self._fov = fov

        # initialize important variables to None
        self._dtype = None
        self._device = None
        self._equi = None
        self._target = None
        self._pers = None
        self._equi_path = None
        self._initial_rotation = None
        self._target_rotation = None

    @classmethod
    def from_config(cls, cfg: Config):
        return cls(
            height=cfg.sim.height,
            width=cfg.sim.width,
            fov=cfg.sim.fov,
            sampling_mode=cfg.sim.sampling_mode,
        )

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def fov(self) -> float:
        return self._fov

    @property
    def dtype(self) -> Union[np.dtype, torch.dtype]:
        assert self._dtype is not None
        return self._dtype

    @property
    def device(self) -> Optional[torch.device]:
        if self._dtype in (np.float32, np.float64):
            return None
        elif self._dtype in (torch.float16, torch.float32, torch.float64):
            assert self._device is not None
            return self._device
        else:
            raise ValueError("device was not initialized")

    @property
    def equi(self) -> Tensor:
        """Make sure that access to equi is always a clone"""
        return copy_tensor(self._equi)

    @property
    def target(self) -> Tensor:
        """Make sure that access to target is always a clone"""
        return copy_tensor(self._target)

    @target.setter
    def target(self, image: Tensor):
        """This is used to change target image (for corruption)"""
        self._target = image

    @property
    def pers(self) -> Tensor:
        """Make sure that access to pers is always a clone"""
        return copy_tensor(self._pers)

    def inititialize_loader(
        self,
        dtype: Dtypes,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """Initialize the loader for equirectangular image

        params:
        - dtype (np.dtype or torch.dtype)
        - device (torch.device): torch.device('cpu')

        returns: None
        """

        if dtype in (torch.float16, torch.float32, torch.float64):
            self.is_torch = True
        elif dtype in (np.float32, np.float64):
            self.is_torch = False
            # FIXME: warn the user that numpy will result in using all the threads
            # unless they specifically change the environment variables
            # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
            # OPENBLAS_NUM_THREADS=4, python ...  for openblas
            # MKL_NUM_THREADS=4, python ...  for mkl
            # OMP_NUM_THREADS=4, python ...  for openmp
        else:
            raise ValueError(f"ERR: input dtype in invalid; {dtype}")

        # initialize params
        self._dtype = dtype
        self._device = device

        if self.is_torch:
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

    def load_episode(
        self,
        equi_path: str,
        initial_rotation: Rots,
        target_rotation: Rots,
    ) -> None:
        """Load a new episode

        params:
        - equi_path (str)
        - initial_rotation
        - target_rotation

        returns: None

        NOTE:
        - rotations are in degrees (not radians)
        """

        assert self._load_func is not None, \
            "ERR: loading function is not initialized"

        if equi_path != self._equi_path:
            # NOTE: only load equi when the equi_path differs
            self._equi = self._load_func(img_path=equi_path)

        # set images
        self._pers = self.sample(rot=initial_rotation)
        self._target = self.sample(rot=target_rotation)

        # keep data
        self._equi_path = equi_path
        self._initial_rotation = initial_rotation
        self._target_rotation = target_rotation

    def sample(self, rot: Rots) -> Tensor:
        """Sample rotated perspective image from equirectangular

        This method is internal; external calls are acceptable
        """
        rad_rot = deg2rad(rot)
        return self.equi2pers(self.equi, rots=rad_rot)

    def move(self, rot: Rots) -> Tensor:
        """Rotate and return a perspective

        - This method is called externally
        - Calling this method keeps a copy of perspective image inside the class

        NOTE: it is important to call `move` instead of `sample` since `move` saves
        sampled output as internal `pers` which can be called from `pers` or `render_pers`
        """
        self._pers = self.sample(rot=rot)
        return self.pers

    def reset(
        self,
        equi_path: str,
        initial_rotation: Rots,
        target_rotation: Rots,
    ) -> Tensor:
        """Reset episode and returns perspective and target images
        """
        self.load_episode(
            equi_path=equi_path,
            initial_rotation=initial_rotation,
            target_rotation=target_rotation,
        )
        return self.pers, self.target

    def render_pers(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (converted for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.pers, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.pers, to_bgr=to_bgr)

    def render_target(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (converted for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.target, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.target, to_bgr=to_bgr)

    def render_equi(self, to_bgr: bool = True) -> np.ndarray:
        """Return view (converted for cv2.imshow)
        """
        if self.is_torch:
            return post_process_for_render_torch(self.equi, to_bgr=to_bgr)
        else:
            return post_process_for_render(self.equi, to_bgr=to_bgr)

    def get_bounding_fov(self, rot: Rots) -> np.ndarray:
        return self.equi2pers.get_bounding_fov(equi=self.equi, rots=deg2rad(rot))

    def __del__(self):
        del self._equi
        del self._target
        del self._pers
        del self.equi2pers

    def close(self) -> None:
        pass
