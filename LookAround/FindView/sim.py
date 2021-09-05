#!/usr/bin/env python3

from functools import partial
from typing import Dict, Union

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
Rots = Dict[str, Union[int, float]]
Dtypes = Union[np.dtype, torch.dtype]


class FindViewSim(object):

    equi: Tensor
    pers: Tensor

    def __init__(
        self,
        equi_path: str,
        initial_rots: Rots,
        is_torch: bool,
        dtype: Dtypes,
        height: int,
        width: int,
        fov: float,
        sampling_mode: str,
    ) -> None:

        self._equi_path = equi_path
        self._initial_rots = initial_rots

        if is_torch:
            self._load_func = partial(
                load2torch,
                dtype=dtype,
                is_cv2=False,
            )
        else:
            self._load_func = partial(
                load2numpy,
                dtype=dtype,
                is_cv2=False
            )

        self.equi2pers = Equi2Pers(
            height=height,
            width=width,
            fov_x=fov,
            skew=0.0,
            z_down=True,
            mode=sampling_mode,
        )

        # initialize
        self.equi = self._load_from_path(equi_path=self._equi_path)
        self.pers = self._sample_pers(rots=self._initial_rots)

    def _load_from_path(
        self,
        equi_path: str,
    ) -> Tensor:
        equi = self._load_func(img_path=equi_path)
        assert isinstance(equi, (np.ndarray, torch.Tensor)), \
            f"ERR: could not load {equi_path} to Tensor"
        return equi

    def _sample_pers(
        self,
        rots: Rots,
    ) -> Tensor:
        return self.equi2pers(self._copy_tensor(self.equi), rots=rots)

    @staticmethod
    def _copy_tensor(
        img: Tensor,
    ) -> Tensor:
        if isinstance(img, np.ndarray):
            return img.copy()
        elif torch.is_tensor(img):
            return img.clone()
        else:
            raise ValueError("ERR: cannot copy tensor")

    def move(self, rots: Rots) -> Tensor:
        """Rotate view and return unrefined view
        """
        self.pers = self._sample_pers(rots=rots)
        return self.pers

    def reset(
        self,
        equi_path: str,
        initial_rots: Rots,
    ) -> Tensor:
        """Reset rotation and change equi image
        """
        self._initial_rots = initial_rots
        if equi_path != self._equi_path:
            # load when path differs (for efficiency)
            self._equi_path = equi_path
            self.equi = self._load_from_path(equi_path=equi_path)
        return self._sample_pers(rots=initial_rots)

    def get_view(self) -> Tensor:
        """Return view (unrefined)
        """
        return self.pers

    def render(self) -> np.ndarray:
        """Return view (refined for cv2.imshow)
        """
        if isinstance(self.pers, np.ndarray):
            return post_process_for_render(self._copy_tensor(self.pers))
        elif torch.is_tensor(self.pers):
            return post_process_for_render_torch(self._copy_tensor(self.pers))
        else:
            raise ValueError("ERR: cannot post process")

    def __del__(self):
        # NOTE: clean up
        del self.equi
        del self.pers
        del self.equi2pers
