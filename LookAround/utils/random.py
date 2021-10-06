#!/usr/bin/env python3

import random

import numpy as np


def seed(n: int = 0):
    random.seed(n)
    np.random.seed(n)
