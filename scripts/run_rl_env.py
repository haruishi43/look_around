#!/usr/bin/env python3

"""Run `RLEnv`

"""

import argparse
from functools import partial
import json
import os
import random
from typing import List

import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.FindView.env import FindViewActions
from LookAround.FindView.rl_env import make_rl_env
from LookAround.utils.visualizations import save_images_as_video

from findview_baselines.agents.greedy import GreedyMovementAgent
from findview_baselines.agents.feature_matching import FeatureMatchingAgent

random.seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


class SingleMovementAgent(Agent):
    def __init__(self, action: str = "right") -> None:
        assert action in FindViewActions.all
        self.action = action

    def act(self, observations):
        return self.action

    def reset(self):
        ...


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--agent",
        required=True,
        type=str,
        choices=['greedy', 'single', 'fm'],
        help="name of the agent"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    cfg = Config.fromfile(config_path)
    cfg.max_steps = 2000
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

    # initialize agent
    if args.agent == "single":
        agent = SingleMovementAgent(action="right")
    elif args.agent == "greedy":
        agent = GreedyMovementAgent(cfg=cfg)
    elif args.agent == "fm":
        agent = FeatureMatchingAgent(cfg=cfg)
    else:
        raise ValueError

    images = []
    obs = rlenv.reset()
    print(obs.keys())
    render = rlenv.render()
    images.append(render['target'])
    images.append(render['pers'])

    for i in tqdm(range(num_steps)):
        action = agent.act(obs)
        obs, reward, done, info = rlenv.step(action)

        # print("reward", reward, action, done)
        pers = rlenv.render()['pers']
        images.append(pers)
        if done:
            if action == "stop":
                print("called stop")
                assert info['called_stop']
            print(">>> next episode!")
            print("saving results")
            save_path = os.path.join('./results/rlenv', f"{info['img_name']}.json")
            with open(save_path, 'w') as f:
                json.dump(info, f, indent=2)

            # NEED TO RESET!
            obs = rlenv.reset()
            render = rlenv.render()
            images.append(render['target'])
            images.append(render['pers'])
            agent.reset()

    save_path = os.path.join('./results/rlenv', f'test_{args.agent}_rl_env.mp4')
    save_images_as_video(images, save_path)
