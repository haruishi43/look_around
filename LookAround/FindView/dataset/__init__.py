#!/usr/bin/env python3

from mycv.utils import Config

from .dynamic_dataset import DynamicDataset
from .static_dataset import StaticDataset
from .episode import Episode

__all__ = [
    "Episode",
    "make_episode",
]


def make_dataset(cfg: Config, split: str):
    if split in ('val', 'test'):
        return StaticDataset(cfg=cfg, split=split)
    elif split == 'train':
        return DynamicDataset(cfg=cfg)
    else:
        raise ValueError(f"ERR: {split} is not a valid split")
