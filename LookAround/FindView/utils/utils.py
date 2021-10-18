#!/usr/bin/env python3

from copy import deepcopy
from typing import Union

import numpy as np
import torch

from LookAround.core.improc import post_process_for_render, post_process_for_render_torch


def obs2img(
    pers: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    to_bgr: bool = False,
) -> np.ndarray:
    """Generate concatenated frame for validation/benchmark videos
    """

    # NOTE: make sure that the operations don't change the observations
    _pers = deepcopy(pers)
    _target = deepcopy(target)

    if torch.is_tensor(_pers) and torch.is_tensor(_target):
        _pers = post_process_for_render_torch(_pers, to_bgr=to_bgr)
        _target = post_process_for_render_torch(_target, to_bgr=to_bgr)
    else:
        _pers = post_process_for_render(_pers, to_bgr=to_bgr)
        _target = post_process_for_render(_target, to_bgr=to_bgr)
    frame = np.concatenate([_pers, _target], axis=1)
    return frame
