#!/usr/bin/env python3

"""Human interaction mode

"""

import os
import time
from typing import Dict

import cv2
import numpy as np

from LookAround.FindView.sim import FindViewSim


def post_process_for_render(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (1, 2, 0))
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


class Action2Rotation(object):

    def __init__(
        self,
        initial_rots,
        inc: float = np.pi / 90,
        pitch_thresh: float = np.pi / 4,
    ) -> None:
        self.rots = initial_rots
        self.inc = inc
        self.pitch_thresh = pitch_thresh

    def convert(
        self,
        action: str,
    ) -> Dict[str, float]:

        pitch = self.rots['pitch']
        yaw = self.rots['yaw']

        if action == "up":
            pitch += self.inc
        elif action == "down":
            pitch -= self.inc
        elif action == "right":
            yaw += self.inc
        elif action == "left":
            yaw -= self.inc

        if pitch >= self.pitch_thresh:
            pitch = self.pitch_thresh
        elif pitch <= -self.pitch_thresh:
            pitch = -self.pitch_thresh
        if yaw > np.pi:
            yaw -= 2 * np.pi
        elif yaw <= -np.pi:
            yaw += 2 * np.pi

        self.rots = {
            "roll": 0.,
            "pitch": pitch,
            "yaw": yaw,
        }

        return self.rots


class RandomAgent(object):

    def __init__(self) -> None:
        self.actions = ["up", "down", "right", "left", "stop"]

    def act(self):
        return np.random.choice(self.actions)

    def reset(self):
        ...


if __name__ == "__main__":

    # initialize data path
    data_root = "./data/pano1024x512/indoor/bedroom"
    img_name = "pano_afvwdfmjeaglsd.jpg"
    img_path = os.path.join(data_root, img_name)

    # params:
    initial_rots = {
        "roll": 0.,
        "pitch": 0.,
        "yaw": 0.,
    }
    num_steps = 1000
    dtype = np.dtype(np.float32)
    height = 256
    width = 256
    fov = 90.0
    sampling_mode = "bilinear"

    # initialize simulator
    a2r = Action2Rotation(initial_rots=initial_rots)
    sim = FindViewSim(
        equi_path=img_path,
        initial_rots=initial_rots,
        is_torch=False,
        dtype=dtype,
        height=height,
        width=width,
        fov=fov,
        sampling_mode=sampling_mode,
    )

    # render first frame
    pers = sim.render()
    cv2.imshow("pers", pers)

    # stats
    times = []

    for i in range(num_steps):
        print(f"Step {i}")

        # change direction `wasd` or exit with `q`
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
        elif k == ord("w"):
            rots = a2r.convert("up")
        elif k == ord("s"):
            rots = a2r.convert("down")
        elif k == ord("a"):
            rots = a2r.convert("left")
        elif k == ord("d"):
            rots = a2r.convert("right")
        else:
            print("unrecongizable button pressed, moving 1 frame")
            rots = a2r.rots

        s = time.time()

        sim.move(rots)

        # render
        pers = sim.render()
        cv2.imshow("pers", pers)

        e = time.time()
        times.append(e - s)

    cv2.destroyAllWindows()
    mean_time = sum(times) / len(times)
    print("mean time:", mean_time)
    print("fps:", 1 / mean_time)
