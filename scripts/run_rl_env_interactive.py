#!/usr/bin/env python3

"""Run `RLEnv` in interactive mode

"""

import argparse
from functools import partial
from typing import List

import cv2
import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.utils.random import seed
from LookAround.FindView import RLEnvRegistry

from findview_baselines.agents.human import Human

seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


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
    print(">>> Config:")
    print(cfg.pretty_text)

    # params:
    split = 'test'
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 5000
    img_names = ["pano_awotqqaapbgcaf", "pano_asxxieiyhiqchw"]
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_names = partial(filter_episodes_by_img_names, names=img_names)

    rlenv = RLEnvRegistry.build(
        cfg.rl_env.name,
        cfg=cfg,
        split=split,
        filter_fn=filter_by_names,
        dtype=dtype,
        device=device,
    )

    # initialize human
    human = Human.from_config(cfg=cfg)

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
