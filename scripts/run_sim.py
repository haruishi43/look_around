#!/usr/bin/env python3

"""Run agent

"""

import os
import time

import cv2
import numpy as np
import torch

from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.rotation_tracker import RotationTracker


class RandomAgent(object):

    def __init__(self) -> None:
        self.actions = ["up", "down", "right", "left", "stop"]

    def act(self):
        return np.random.choice(self.actions)

    def reset(self):
        ...


if __name__ == "__main__":

    # initialize data path
    data_root = "./data/sun360/indoor/bedroom"
    img_name = "pano_afvwdfmjeaglsd.jpg"
    img_path = os.path.join(data_root, img_name)

    # params:
    initial_rots = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }
    target_rots = {
        "roll": 0,
        "pitch": 20,
        "yaw": -40,
    }
    num_steps = 100
    dtype = torch.float32
    height = 256
    width = 256
    fov = 90.0
    sampling_mode = "bilinear"

    # initialize agent
    agent = RandomAgent()

    # initialize simulator
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode=sampling_mode,
    )
    sim.inititialize_loader(
        dtype=dtype,
        device=torch.device('cpu'),
    )
    sim.reset(
        equi_path=img_path,
        initial_rotation=initial_rots,
        target_rotation=target_rots,
    )

    # add rotation tracker
    rot_tracker = RotationTracker(
        inc=1,
        pitch_threshold=60,
    )
    rot_tracker.reset(initial_rots)

    # render first frame
    target = sim.render_target()
    pers = sim.render_pers()
    # cv2.imshow("pers", pers)

    # stats
    times = []

    for i in range(num_steps):
        print(f"Step {i}")
        k = cv2.waitKey(1)

        action = agent.act()
        rots = rot_tracker.move(action)

        s = time.time()

        sim.move(rots)

        # render
        pers = sim.render_pers()
        # cv2.imshow("pers", pers)

        e = time.time()
        times.append(e - s)

    # cv2.destroyAllWindows()
    mean_time = sum(times) / len(times)
    print("mean time:", mean_time)
    print("fps:", 1 / mean_time)
