#!/usr/bin/env python3

"""Simulator Spec

"""

from functools import partial
from typing import Union

import numpy as np

import torch

from .improc import load2numpy, load2torch


class BaseSim(object):

    def __init__(self) -> None:
        ...
