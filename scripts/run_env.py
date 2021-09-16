#!/usr/bin/env python3

"""Run `Env`

Before building parallel environments, we need to test the basic environments

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
from LookAround.FindView.env import FindViewActions, make_env
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

    def act(self, observation):
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
    split = 'test'
    is_torch = True
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 5000
    img_names = ["pano_awotqqaapbgcaf", "pano_asxxieiyhiqchw"]
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_names = partial(filter_episodes_by_img_names, names=img_names)

    # initialize env
    env = make_env(
        cfg=cfg,
        split=split,
        filter_fn=filter_by_names,
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

    obs = env.reset()
    print(obs.keys())
    render = env.render()
    images.append(render['target'])
    images.append(render['pers'])

    for i in tqdm(range(num_steps)):
        action = agent.act(obs)
        obs = env.step(action)
        pers = env.render()['pers']
        images.append(pers)
        if env.episode_over:
            print("next episode!")
            # save stats to file
            stats = env.get_metrics()
            img_name = env.current_episode.img_name
            save_path = os.path.join('./results/env', f'{img_name}.json')
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2)

            # NEED TO RESET!
            obs = env.reset()
            render = env.render()
            images.append(render['target'])
            images.append(render['pers'])
            agent.reset()

    save_path = os.path.join('./results/env', f'{args.agent}_env.mp4')
    save_images_as_video(images, save_path)
