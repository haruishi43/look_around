#!/usr/bin/env python3

"""
Trying out feature matching on OpenCV for feature matching agent

Input is the target image and a close image

- Compute feature matching
- Get coordinates of matched feature
- Compute population densities (where is it matching)
- Decide action (Move in yaw or pitch direction or stop)

"""

import cv2