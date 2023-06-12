#!/usr/bin/env python3

"""Testing out parallel Env

"""

import argparse
import json
import os
import random
from typing import List

# import numpy as np
import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.FindView.rl_env import FindViewRLEnv
from LookAround.FindView.vec_env import construct_envs
from LookAround.utils.visualizations import images_to_video_cv2

from findview_baselines.agents.single_movement import SingleMovementAgent
from findview_baselines.agents.greedy import GreedyMovementAgent
from findview_baselines.agents.feature_matching import FeatureMatchingAgent

random.seed(0)


def filter_episodes_by_img_names(episode, names: List[str]) -> bool:
    return episode.img_name in names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset",
    )
    parser.add_argument(
        "--agent",
        required=True,
        type=str,
        choices=["single", "greedy", "fm"],
        help="name of the agent",
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
    split = "train"
    dtype = torch.float32
    device = torch.device("cpu")
    num_steps = 500

    envs = construct_envs(
        cfg=cfg,
        split=split,
        is_rlenv=True,
        dtype=dtype,
        device=device,
        vec_type=vec_type,
    )

    # initialize agent
    if args.agent == "single":
        agents = [
            SingleMovementAgent.from_config(cfg=cfg)
            for _ in range(cfg.num_envs)
        ]
    elif args.agent == "greedy":
        agents = [
            GreedyMovementAgent.from_config(cfg=cfg)
            for _ in range(cfg.num_envs)
        ]
    elif args.agent == "fm":
        agents = [
            FeatureMatchingAgent.from_config(cfg=cfg)
            for _ in range(cfg.num_envs)
        ]
    else:
        raise ValueError

    assert envs.num_envs == cfg.num_envs

    images = []

    # reset env
    print("reset!")
    obs = envs.reset()
    print(len(obs), obs[0].keys())
    render = envs.render()
    # images.append(render['target'])
    images.append(render["pers"])
    episodes = envs.current_episodes()

    for step in tqdm(range(num_steps)):
        actions = [agent.act() for agent in agents]

        outputs = envs.step(actions)
        obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        # print(rewards)
        # print(dones)
        pers = envs.render()["pers"]
        # target = envs.render()['target']
        images.append(pers)

        for i, done in enumerate(dones):
            if done:
                if actions[i] == "stop":
                    print(f"{i} called stop")
                    assert infos[i]["called_stop"]
                print(f"Loading next episode for {i}")
                save_path = os.path.join(
                    "./results/vecrlenv",
                    f"{args.agent}_{i}_{infos[i]['img_name']}.json",
                )
                with open(save_path, "w") as f:
                    json.dump(infos[i], f, indent=2)

                agents[i].reset()

    images_to_video_cv2(
        images=images,
        output_dir="./results/vecrlenv",
        video_name=f"{args.agent}_test",
        fps=30.0,
    )
