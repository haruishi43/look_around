#!/usr/bin/env python3

from mycv.utils import Config

from .dynamic_dataset import DynamicDataset
from .static_dataset import StaticDataset
from .episode import Episode, PseudoEpisode

__all__ = [
    "Episode",
    "PseudoEpisode",
    "make_episode",
]


def make_dataset(cfg: Config, split: str, filter_fn=None):
    if split in ('val', 'test'):
        dataset = StaticDataset(cfg=cfg, split=split)
    elif split == 'train':
        dataset = DynamicDataset(cfg=cfg)
    else:
        raise ValueError(f"ERR: {split} is not a valid split")

    if filter_fn is not None:
        dataset = dataset.filter_dataset(filter_fn)

    return dataset
