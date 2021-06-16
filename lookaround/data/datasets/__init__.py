#!/usr/bin/env python3

from .registry import DATASET_REGISTRY

# NOTE: import all datasets here
from . import sun360  # ensure initialization

__all__ = [k for k in globals().keys() if not k.startswith("_")]
