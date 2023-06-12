#!/usr/bin/env python3

import os
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ],
)

to_PIL = transforms.Compose(
    [
        transforms.ToPILImage(),
    ]
)


def post_process_for_render(img: np.ndarray, to_bgr: bool = True) -> np.ndarray:
    img = np.transpose(img, (1, 2, 0))
    img *= 255
    img = img.astype(np.uint8)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def post_process_for_render_torch(
    img: torch.Tensor, to_bgr: bool = True
) -> np.ndarray:
    img = img.to("cpu")
    img *= 255
    img = img.type(torch.uint8)
    img = img.numpy()
    img = img.transpose((1, 2, 0))
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _open_as_PIL(img_path: str) -> Image.Image:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    img = Image.open(img_path)
    assert img is not None
    if img.getbands() == tuple("RGBA"):
        # NOTE: Sometimes images are RGBA
        img = img.convert("RGB")
    return img


def _open_as_cv2(img_path: str) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    # FIXME: shouldn't use `imread` since it won't auto detect color space
    warnings.warn("Cannot handle color spaces other than RGB")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    assert img is not None
    return img


def load2numpy(
    img_path: str,
    dtype: np.dtype,
    is_cv2: bool = False,
) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)
        img = np.asarray(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img = np.transpose(img, (2, 0, 1))

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0
    dist_dtype = np.dtype(dtype)
    if dist_dtype in (np.float32, np.float64):
        img = img / 255.0
    img = img.astype(dist_dtype)

    return img


def load2torch(
    img_path: str,
    dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
    is_cv2: bool = False,
) -> torch.Tensor:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0 (ToTensor)
    if dtype in (torch.float16, torch.float32, torch.float64):
        img = to_tensor(img)
        # FIXME: force typing since I have no idea how to change types in
        # PIL; also it's easier to change type using `type`; might be slower
        img = img.type(dtype)
        # NOTE: automatically adds channel for grayscale
    elif dtype == torch.uint8:
        img = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True))
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute((2, 0, 1)).contiguous()
        assert img.dtype == torch.uint8

    img = img.to(device)
    return img
