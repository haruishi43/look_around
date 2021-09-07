#!/usr/bin/env python3

"""Running `Sim`

I think there're generally two important parts in an Env
- `Sim`
- `Rotation`: Converting rotation to action

NOTE: think about how to make the `Sim` Parallel!

"""

import argparse
from functools import partial
from typing import Dict, List

from tqdm import tqdm

from mycv.utils import Config
import numpy as np
import torch

from LookAround.FindView.dataset import Episode, make_dataset
from LookAround.FindView.sim import FindViewSim


def filter_episodes_by_img_names(episode: Episode, names: List[str]) -> bool:
    return episode.img_name in names


def filter_episodes_by_sub_labels(episode: Episode, sub_labels: List[str]) -> bool:
    return episode.sub_label in sub_labels


class Action2Rotation(object):

    def __init__(
        self,
        initial_rot,
        inc: int = 1,
        pitch_thresh: int = 60,
    ) -> None:
        self.rot = initial_rot
        self.inc = inc
        self.pitch_thresh = pitch_thresh

    def convert(
        self,
        action: str,
    ) -> Dict[str, float]:

        pitch = self.rot['pitch']
        yaw = self.rot['yaw']

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
        if yaw > 180:
            yaw -= 2 * 180
        elif yaw <= -180:
            yaw += 2 * 180

        self.rots = {
            "roll": 0,
            "pitch": pitch,
            "yaw": yaw,
        }

        return {
            "roll": 0.,
            "pitch": pitch * np.pi / 180,
            "yaw": yaw * np.pi / 180,
        }


class SingleMovementAgent(object):
    def __init__(self, action: str = "right") -> None:
        self.action = action

    def act(self):
        return self.action


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

    # params
    split = 'test'
    img_names = ["pano_awotqqaapbgcaf", "pano_asxxieiyhiqchw"]
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_names = partial(filter_episodes_by_img_names, names=img_names)
    filter_by_labels = partial(filter_episodes_by_sub_labels, sub_labels=sub_labels)

    # initialize dataset
    dataset = make_dataset(cfg=cfg, split=split)
    dataset = dataset.filter_episodes(filter_by_names)
    print(f"Using {len(dataset)}")  # 200
    episode_iterator = dataset.get_iterator(
        cycle=True,
        shuffle=False,
        num_episode_sample=30,
    )

    # initialze agent
    agent = SingleMovementAgent(action="right")

    # initialize sim
    sim = FindViewSim(
        **cfg.sim,
    )
    sim.inititialize_loader(
        is_torch=True,
        dtype=torch.float32,
        device=torch.device('cuda:0'),
    )
    # sim.inititialize_loader(
    #     is_torch=True,
    #     dtype=torch.float32,
    #     device=torch.device('cpu'),
    # )
    # sim.inititialize_loader(
    #     is_torch=False,
    #     dtype=np.float32,
    # )
    num_iter = 10
    num_steps = 1000
    for i in range(num_iter):
        episode = next(episode_iterator)
        print(episode.img_name)

        # reset/init sim from episode
        pers, target = sim.reset(
            equi_path=episode.path,
            initial_rotation=episode.initial_rotation,
            target_rotation=episode.target_rotation,
        )

        # initialize action2rotation conversion
        a2r = Action2Rotation(
            initial_rot=episode.initial_rotation,
            inc=cfg.step_size,
            pitch_thresh=cfg.pitch_threshold,
        )

        for j in tqdm(range(num_steps)):
            action = agent.act()
            rot = a2r.convert(action)
            pers = sim.move(rot)

            render_pers = sim.render_pers()
