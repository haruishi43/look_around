#!/usr/bin/env python3

"""Run `RLEnv`

"""

import argparse
from functools import partial
import random
from typing import List

import cv2
import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.FindView.env import FindViewActions
from LookAround.FindView.rl_env import make_rl_env

random.seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


class Human(object):

    def __init__(self):
        ...

    def reset(self):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    cfg = Config.fromfile(config_path)
    cfg.max_steps = 2500
    print(">>> Config:")
    print(cfg.pretty_text)

    # params:
    split = 'train'
    is_torch = True
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 5000
    img_names = ["pano_awotqqaapbgcaf", "pano_asxxieiyhiqchw"]
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_names = partial(filter_episodes_by_img_names, names=img_names)

    rlenv = make_rl_env(
        cfg=cfg,
        split=split,
        # filter_fn=filter_by_names,
        is_torch=is_torch,
        dtype=dtype,
        device=device,
    )

    # initialize human
    human = Human()

    obs = rlenv.reset()

    # render
    render = rlenv.render()
    cv2.imshow("target", render['target'])
    cv2.imshow("pers", render['pers'])

    steps = 0
    for i in tqdm(range(num_steps)):  # replace it with `while True`

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
        obs, reward, done, info = rlenv.step(action)

        # print debug information
        print("reward", reward, action, done)

        # render
        pers = rlenv.render()['pers']
        cv2.imshow("pers", pers)

        # update step
        steps += 1

        if done:
            if action == "stop":
                print("called stop")
                assert info['called_stop']
            print(">>> next episode!")

            # NEED TO RESET!
            obs = rlenv.reset()
            render = rlenv.render()
            human.reset()

            steps = 0
