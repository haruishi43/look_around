#!/usr/bin/env python3

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
    if torch.is_tensor(pers) and torch.is_tensor(target):
        _pers = post_process_for_render_torch(pers, to_bgr=to_bgr)
        _target = post_process_for_render_torch(target, to_bgr=to_bgr)
    else:
        _pers = post_process_for_render(pers, to_bgr=to_bgr)
        _target = post_process_for_render(target, to_bgr=to_bgr)
    frame = np.concatenate([_pers, _target], axis=1)
    return frame
