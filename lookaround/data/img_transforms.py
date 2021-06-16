#!/usr/bin/env python3

import logging
import random
from typing import List, Optional, Tuple, Union

import torch

from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)

from pers2pano.config import CfgNode, configurable

__all__ = [
    "build_transforms",
    "build_untransform",
]


class ColorAugmentation:
    """Randomly alters the intensities of RGB channels.
    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.
    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomShuffleWidth:
    """Random Shuffle Width

    Randomly shuffle the tensor in width direction
    """

    def __init__(self):
        pass

    def __call__(self, tensor: torch.Tensor):
        w = tensor.shape[-1]
        x = random.randint(0, w)
        b, e = torch.split(tensor, [x, w - x], dim=-1)
        tensor = torch.cat((e, b), dim=-1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(norm_mean={}, norm_std={})".format(
            self.mean, self.std
        )


def _from_cfg(cfg: CfgNode):
    transform_cfg = getattr(cfg, "transforms")
    return {
        "height": transform_cfg.pano_height,
        "width": transform_cfg.pano_width,
        "transforms": transform_cfg.transforms,
        "norm_mean": transform_cfg.norm_mean,
        "norm_std": transform_cfg.norm_std,
    }


@configurable(from_config=_from_cfg)
def build_transforms(
    height: int,
    width: int,
    transforms: Optional[Union[List[str], str]] = None,
    norm_mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
    norm_std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
    **kwargs,
):
    """Builds train, test, visualize transform functions.
    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.

    NOTE:

    ImageNet values
        norm_mean = (0.485, 0.456, 0.406)
        norm_std  = (0.229, 0.224, 0.225)

    GAN values
        norm_mean = (0.5, 0.5, 0.5)
        norm_std  = (0.5, 0.5, 0.5)
    """

    logger = logging.getLogger("pers2pano")

    if transforms is None or len(transforms) == 0:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            f"transforms must be a list of strings, but found to be {transforms}"
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    logger.info("Building transforms ...")
    transform_train = []
    transform_test = []

    logger.info(f"+ resize to {height}x{width}")
    transform_train += [Resize((height, width))]
    transform_test += [Resize((height, width))]

    if "random_flip" in transforms:
        logger.info("+ random flip")
        transform_train += [RandomHorizontalFlip()]

    if "color_jitter" in transforms:
        logger.info("+ color jitter")
        transform_train += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    logger.info("+ to torch tensor of range [0, 1]")
    transform_train += [ToTensor()]
    transform_test += [ToTensor()]

    if "normalize" in transforms:
        logger.info(f"+ normalization (mean={norm_mean}, std={norm_std})")
        transform_train += [Normalize(mean=norm_mean, std=norm_std)]
        transform_test += [Normalize(mean=norm_mean, std=norm_std)]

    # FIXME: Add more augmentations
    if "color_augmentation" in transforms:
        logger.info("+ color augmentation")
        transform_train += [ColorAugmentation()]

    if "random_shuffle" in transforms:
        logger.info("+ random shuffle")
        transform_train += [RandomShuffleWidth()]

    transform_train = Compose(transform_train)
    transform_test = Compose(transform_test)

    return transform_train, transform_test


@configurable(from_config=_from_cfg)
def build_untransform(
    transforms: Optional[Union[List[str], str]] = None,
    norm_mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
    norm_std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
    **kwargs,
):
    if transforms is None or len(transforms) == 0:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    untransforms = []

    if "normalize" in transforms:
        untransforms += [UnNormalize(mean=norm_mean, std=norm_std)]
    untransforms += [ToPILImage()]
    untransform = Compose(untransforms)
    return untransform
