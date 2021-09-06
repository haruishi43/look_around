#!/usr/bin/env python3

"""Initializing Dataset for use

Where is it used?
- Env

Uses:
- Training
- Benchmarking
- Simple Evaluation

FIXME: when there are multiple datasets, lets add `registry`
"""

from mycv.utils import Config

from .dynamic_dataset import DynamicDataset
from .static_dataset import StaticDataset


def make_dataset(cfg: Config, split: str):
    if split in ('val', 'test'):
        return StaticDataset(cfg=cfg, split=split)
    elif split == 'train':
        return DynamicDataset(cfg=cfg)
    else:
        raise ValueError(f"ERR: {split} is not a valid split")
