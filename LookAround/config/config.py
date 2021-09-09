#!/usr/bin/env python3

try:
    from mycv.utils import Config, DictAction
except ImportError:
    from mmcv.utils import Config, DictAction

__all__ = [
    "Config",
    "DictAction",
]
