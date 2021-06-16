#!/usr/bin/env python3

from lookaround.data.datasets.dataset import Dataset
from lookaround.utils.registry import Registry

__all__ = ["DATASET_REGISTRY"]

DATASET_REGISTRY = Registry(
    name="DATASET",
    inherits=Dataset,
)
