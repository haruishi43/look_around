#!/usr/bin/env python3

try:
    from mycv.utils import Registry
except ImportError:
    from mmcv.utils import Registry

__all__ = ["Registry"]
