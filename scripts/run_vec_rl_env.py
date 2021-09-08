#!/usr/bin/env python3

"""Testing out parallel Env

"""

import argparse
import json
import os
import random
from typing import List

from mycv.utils import Config
import numpy as np
import torch
from tqdm import tqdm

from LookAround.FindView.env import FindViewActions, FindViewRLEnv, FindViewEnv
from LookAround.FindView.vec_env import construct_envs
from LookAround.utils.visualizations import save_images_as_video

random.seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


class SingleMovementAgent(object):
    def __init__(self, action: str = "right") -> None:
        assert action in FindViewActions.all
        self.action = action

    def act(self):
        return self.action

    def reset(self):
        ...


def movement_generator(size=4):
    idx = 0
    repeat = 1
    while True:
        for r in range(repeat):
            yield idx

        idx = (idx + 1) % size
        if idx % 2 == 0:
            repeat += 1


class GreedyMovementAgent(object):
    def __init__(self, chance=0.001) -> None:
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        self.stop_chance = chance
        for action in self.movement_actions:
            assert action in FindViewActions.all
        self.g = movement_generator(len(self.movement_actions))

    def act(self):
        if random.random() < self.stop_chance:
            return self.stop_action
        return self.movement_actions[next(self.g)]

    def reset(self):
        self.g = movement_generator(len(self.movement_actions))


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
    vec_type = "threaded"
    env_cls = FindViewRLEnv
    split = 'train'
    is_torch = False
    dtype = np.float32
    device = torch.device('cuda:0')
    num_steps = 500

    envs = construct_envs(
        env_cls=env_cls,
        cfg=cfg,
        split=split,
        is_torch=is_torch,
        dtype=dtype,
        device=device,
        vec_type=vec_type,
    )

    agents = [GreedyMovementAgent() for _ in range(cfg.num_envs)]

    assert envs.num_envs == cfg.num_envs

    images = []

    # reset env
    print("reset!")
    obs = envs.reset()
    print(len(obs), obs[0].keys())
    render = envs.render()
    # images.append(render['target'])
    images.append(render['pers'])
    episodes = envs.current_episodes()

    for step in tqdm(range(num_steps)):

        actions = [agent.act() for agent in agents]

        outputs = envs.step(actions)
        obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        # print(rewards)
        # print(dones)
        pers = envs.render()['pers']
        images.append(pers)

        for i, done in enumerate(dones):
            if done:
                if actions[i] == "stop":
                    print(f"{i} called stop")
                    assert infos[i]['called_stop']
                print(f"Loading next episode for {i}")
                save_path = os.path.join('./results/vecrlenv', f"{i}_{infos[i]['img_name']}.json")
                with open(save_path, 'w') as f:
                    json.dump(infos[i], f, indent=2)

                agents[i].reset()

    save_path = os.path.join('./results/vecrlenv', 'test.mp4')
    save_images_as_video(images, save_path)
