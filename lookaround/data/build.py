#!/usr/bin/env python3

"""Build functions (kinda hard-coded, but you know...)
"""

import os.path as osp
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from lookaround.config import CfgNode
from lookaround.data.img_transforms import build_transforms
from lookaround.data.datasets.dataset import Dataset
from lookaround.data.datasets.registry import DATASET_REGISTRY
from lookaround.data.samplers import build_sampler

__all__ = [
    "build_dataloaders",
    "create_dataloader",
]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler: Optional[Sampler] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[Callable] = None,
    use_gpu: bool = True,
) -> DataLoader:
    """Create a dataloader with Dataset and params"""
    use_gpu = torch.cuda.is_available() and use_gpu

    if sampler is not None:
        # FIXME: force mutally exclusive
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader


def build_dataloaders(cfg: CfgNode) -> Tuple[DataLoader]:
    """Build train, val, test dataloaders from CfgNode"""

    train_transforms, test_transforms = build_transforms(cfg)

    loaders = []
    for mode in ("train", "val", "test"):
        transforms = train_transforms if mode == "train" else test_transforms

        dataset_cfg = getattr(cfg, cfg.dataset)

        dataset = DATASET_REGISTRY.get(cfg.dataset)(
            root_path=osp.join(cfg.data_root, dataset_cfg.root),
            data_path=getattr(dataset_cfg, f"{mode}_path"),
            img_transforms=transforms,
        )

        loader_cfg = getattr(getattr(cfg, cfg.task), f"{mode}_loader")

        if mode == "train":
            sampler = build_sampler(
                data_source=dataset,
                train_sampler=loader_cfg.sampler,
                batch_size=loader_cfg.batch_size,
                num_instances=loader_cfg.num_instances,
            )
        else:
            sampler = None

        loader = create_dataloader(
            dataset=dataset,
            batch_size=loader_cfg.batch_size,
            num_workers=loader_cfg.num_workers,
            sampler=sampler,
            shuffle=loader_cfg.shuffle,
            drop_last=loader_cfg.drop_last,
            use_gpu=cfg.use_gpu,
        )
        loaders.append(loader)
    assert len(loaders) == 3

    return loaders
