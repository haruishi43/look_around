#!/usr/bin/env python3

from typing import Union

import numpy as np

import torch


class Equi2Pers:
    """Wrapper for cropping panorama image to perspective image

    FIXME: what is this wrapper for?
    """
    def __init__(
        self,
        w_pers: int,
        h_pers: int,
        fov_x: Union[float, int],
        skew: float = 0.0,
        sampling_method: str = "default",
        mode: str = "bilinear",
        z_down: bool = True,
    ) -> None:

        from equilib import Equi2Pers

        self.equi2pers = Equi2Pers(
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )

    def __call__(
        self,
        rad,
        equi: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        pers = self.equi2pers(equi)
        return pers
