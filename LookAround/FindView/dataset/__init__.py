#!/usr/bin/env python3

from typing import Callable, Optional, Union

from LookAround.config import Config
from LookAround.FindView.dataset.dynamic_dataset import DynamicDataset
from LookAround.FindView.dataset.static_dataset import StaticDataset
from LookAround.FindView.dataset.episode import Episode, PseudoEpisode

__all__ = [
    "Episode",
    "PseudoEpisode",
    "make_episode",
]


def make_dataset(
    cfg: Config,
    split: str,
    filter_fn: Optional[Callable[..., bool]] = None,
    num_episodes_per_img: int = -1,
) -> Union[DynamicDataset, StaticDataset]:
    """Simple function to create dataset from config

    params:
    - cfg (Config)
    - split (str): ('train', 'val', 'test')
    - filter_fn (Callable): None
    """

    # initialize dataset
    if split in ("val", "test"):
        dataset = StaticDataset.from_config(cfg=cfg, split=split)
    elif split == "train":
        dataset = DynamicDataset.from_config(cfg=cfg, split="train")
    else:
        raise ValueError(f"ERR: {split} is not a valid split")

    # filter out episodes/pseudos
    if filter_fn is not None:
        dataset = dataset.filter_dataset(filter_fn)

    if num_episodes_per_img > 0 and isinstance(dataset, StaticDataset):
        dataset = dataset.reduce_dataset(
            num_episodes_per_img=num_episodes_per_img
        )

    return dataset
