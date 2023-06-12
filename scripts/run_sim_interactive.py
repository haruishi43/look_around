#!/usr/bin/env python3

"""Human interaction mode

"""

from copy import deepcopy
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from LookAround.FindView.actions import FindViewActions
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.rotation_tracker import RotationTracker
from LookAround.utils.visualizations import images_to_video_cv2
from LookAround.FindView.utils.visualize import generate_movement_video


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


def rot2coord(rot: dict, h: int, w: int):
    pitch = rot["pitch"]
    yaw = rot["yaw"]

    pitch = -pitch + 90
    yaw = yaw + 180

    y = int(h * pitch / 180 + 0.5)
    x = int(w * yaw / 360 + 0.5)

    y = np.clip(y, 0, h)
    x = np.clip(x, 0, w)
    return (x, y)


def draw_movements(
    equi: np.ndarray,
    history: list,
    target: dict,
) -> np.ndarray:
    # params:
    radius = 8
    thickness = -1

    h, w, c = equi.shape
    history_size = len(history)  # used for color

    img = equi  # deepcopy?

    # initial
    color = (255, 255, 255)
    coord = rot2coord(history[0], h, w)
    img = cv2.circle(img, coord, radius, color, thickness)

    # add history
    for i, rot in enumerate(history):
        color = (
            127,
            int(np.clip(255 * (i + 1) / history_size, 0, 255)),
            127,
            # int(np.clip(255 * (history_size - i) / history_size, 0, 255)),
        )
        coord = rot2coord(rot, h, w)
        img = cv2.circle(img, coord, radius, color, thickness)

    # add target
    color = (0, 0, 0)
    coord = rot2coord(target, h, w)
    img = cv2.circle(img, coord, radius, color, thickness)

    return img


if __name__ == "__main__":
    # initialize data path
    save_root = "./results/interactive/"
    data_root = "./data/sun360/indoor/bedroom"
    img_name = "pano_abcdnxnujvcavm"
    img_with_ext = f"{img_name}.jpg"
    img_path = os.path.join(data_root, img_with_ext)

    # params:
    will_write = False
    is_video = False
    initial_rots = {
        "roll": 0,
        "pitch": 50,
        "yaw": 0,
    }
    target_rots = {
        "roll": 0,
        "pitch": -20,
        "yaw": 180,
    }
    num_steps = 2000
    dtype = torch.float32
    height = 256
    width = 256
    fov = 90.0
    sampling_mode = "bilinear"

    # initialize simulator
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode=sampling_mode,
    )
    sim.inititialize_loader(
        dtype=dtype,
        device=torch.device("cpu"),
    )
    sim.reset(
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
    rot_tracker.reset(initial_rots)

    # show target image
    target = sim.render_target()
    cv2.imshow("target", target)
    if will_write:
        cv2.imwrite(os.path.join(save_root, f"{img_name}_target.jpg"), target)

    # render first frame
    _pers = sim.render_pers()
    cv2.imshow("pers", _pers)

    # stats
    times = []
    pers = [_pers]
    actions = []

    steps = 0
    for i in range(num_steps):
        print(f"Step: {steps}")

        # change direction `wasd` or exit with `q`
        k = cv2.waitKey(0)
        if k == 27:
            print("Exiting")
            break
        elif k not in (ord("w"), ord("s"), ord("a"), ord("d"), ord("q")):
            print("unrecongizable button pressed, moving 1 frame")
            continue

        action = human.act(k)
        actions.append(action)
        if action == "stop":
            break

        rots = rot_tracker.move(action)

        s = time.time()
        sim.move(rots)
        # render
        _pers = sim.render_pers()
        cv2.imshow("pers", _pers)
        pers.append(_pers)

        e = time.time()
        times.append(e - s)

        steps += 1

    cv2.destroyAllWindows()
    mean_time = sum(times) / len(times)
    print("mean time:", mean_time)
    print("fps:", 1 / mean_time)

    if will_write:
        if not is_video:
            for i, frame in enumerate(pers):
                cv2.imwrite(
                    os.path.join(save_root, f"{img_name}_{i}.jpg"), frame
                )
        else:
            images_to_video_cv2(
                images=pers,
                output_dir=save_root,
                video_name=f"{img_name}_video",
                fps=30.0,
            )

    history = rot_tracker.history

    # save history
    # print(history)
    print("last/target:", history[-1], target_rots)
    equi = sim.render_equi()
    img = draw_movements(equi, deepcopy(history), target_rots)
    cv2.imwrite(os.path.join(save_root, f"{img_name}_path.jpg"), img)

    # save movements
    pers_bboxs = []
    for rot in tqdm(history):
        pers_bboxs.append(sim.get_bounding_fov(rot))
    target_bbox = sim.get_bounding_fov(target_rots)

    frames = generate_movement_video(
        output_dir=save_root,
        video_name=f"{img_name}_pretty",
        equi=sim.render_equi(to_bgr=True),
        pers=pers,
        target=target,
        pers_bboxs=pers_bboxs,
        target_bbox=target_bbox,
        actions=actions,
        use_imageio=False,
        fps=30,
    )
