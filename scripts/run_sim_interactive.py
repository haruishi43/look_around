#!/usr/bin/env python3

"""Human interaction mode

"""

import os
import time

import cv2
import numpy as np
import torch

from LookAround.FindView.actions import FindViewActions
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.rotation_tracker import RotationTracker


class Human(object):

    def __init__(self):
        ...

    def act(self, k):
        if k == ord("w"):
            ret = "up"
        elif k == ord("s"):
            ret = "down"
        elif k == ord("a"):
            ret = "left"
        elif k == ord("d"):
            ret = "right"
        elif k == ord("q"):
            ret = "stop"
        else:
            raise ValueError(f"Pressed {k}")

        assert ret in FindViewActions.all
        return ret


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

    # initialize simulator
    # initialize simulator
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode=sampling_mode,
    )
    sim.inititialize_loader(
        is_torch=True,
        dtype=dtype,
        device=torch.device('cpu'),
    )
    sim.initialize_from_episode(
        equi_path=img_path,
        initial_rotation=initial_rots,
        target_rotation=target_rots,
    )

    human = Human()

    # add rotation tracker
    rot_tracker = RotationTracker(
        inc=1,
        pitch_threshold=60,
    )
    rot_tracker.initialize(initial_rots)

    # show target image
    target = sim.render_target()
    cv2.imshow("target", target)
    # render first frame
    pers = sim.render_pers()
    cv2.imshow("pers", pers)

    # stats
    times = []

    steps = 0
    for i in range(num_steps):
        print(f"Step: {steps}")

        # change direction `wasd` or exit with `q`
        k = cv2.waitKey(1)
        if k == 27:
            print("Exiting")
            break
        elif k not in (ord("w"), ord("s"), ord("a"), ord("d"), ord("q")):
            print("unrecongizable button pressed, moving 1 frame")
            continue

        action = human.act(k)
        rots = rot_tracker.convert(action)

        s = time.time()
        sim.move(rots)
        # render
        pers = sim.render()
        cv2.imshow("pers", pers)

        e = time.time()
        times.append(e - s)

        steps += 1

    cv2.destroyAllWindows()
    mean_time = sum(times) / len(times)
    print("mean time:", mean_time)
    print("fps:", 1 / mean_time)
