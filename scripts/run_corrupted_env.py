#!/usr/bin/env python3

"""Run `CorruptedFindViewEnv`

Before building parallel environments, we need to test the basic environments
"""

import argparse
from functools import partial
import json
import os
from typing import List

import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.utils.random import seed
from LookAround.FindView.corrupted_env import CorruptedFindViewEnv
from LookAround.utils.visualizations import images_to_video_cv2, renders_to_image

from findview_baselines.agents.single_movement import SingleMovementAgent
from findview_baselines.agents.greedy import GreedyMovementAgent
from findview_baselines.agents.feature_matching import FeatureMatchingAgent

seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


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
        choices=['single', 'greedy', 'fm'],
        help="name of the agent"
    )
    parser.add_argument(
        "--name",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--corruption",
        default='all',
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    cfg = Config.fromfile(config_path)
    cfg.dataset.max_steps = 250
    print(">>> Config:")
    print(cfg.pretty_text)

    # params:
    split = 'test'
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 1000

    # indoor specific
    img_names = ["pano_awotqqaapbgcaf", "pano_asxxieiyhiqchw"]
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_names = partial(filter_episodes_by_img_names, names=img_names)

    if args.corruption:
        cfg.corrupter.corruptions = args.corruption

    # initialize env
    env = CorruptedFindViewEnv.from_config(
        cfg=cfg,
        split=split,
        filter_fn=filter_by_names,
        dtype=dtype,
        device=device,
    )
    # initialize agent
    if args.agent == "single":
        agent = SingleMovementAgent.from_config(cfg)
    elif args.agent == "greedy":
        agent = GreedyMovementAgent.from_config(cfg)
    elif args.agent == "fm":
        agent = FeatureMatchingAgent.from_config(cfg)
    else:
        raise ValueError

    images = []

    obs = env.reset()
    print(obs.keys())
    render = env.render()
    images.append(renders_to_image(render))

    for i in tqdm(range(num_steps)):
        action = agent.act(obs)
        obs = env.step(action)
        render = env.render()
        images.append(renders_to_image(render))
        if env.episode_over:
            print("next episode!")
            # save stats to file
            stats = env.get_metrics()
            img_name = env.current_episode.img_name
            save_path = os.path.join('./results/tests/corrupted_env', f'{img_name}_{args.name}.json')
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2)

            # NEED TO RESET!
            obs = env.reset()
            render = env.render()
            images.append(renders_to_image(render))
            agent.reset()

    print("fames", len(images))

    images_to_video_cv2(
        images=images,
        output_dir='./results/tests/corrupted_env',
        video_name=f'{args.agent}_{args.name}',
        fps=30.0,
    )
