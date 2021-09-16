#!/usr/bin/env python3

"""Run `Sim`

I think there're generally two important parts in an Env
- `Sim`
- `Rotation`: Converting rotation to action

NOTE: think about how to make the `Sim` Parallel!

"""

import argparse
from functools import partial
import os
from typing import List

from tqdm import tqdm

import torch

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.FindView.dataset import Episode, make_dataset
from LookAround.FindView.env import FindViewActions
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.rotation_tracker import RotationTracker
from LookAround.utils.visualizations import save_images_as_video

from findview_baselines.agents.greedy import GreedyMovementAgent
from findview_baselines.agents.feature_matching import FeatureMatchingAgent


def filter_episodes_by_sub_labels(episode: Episode, sub_labels: List[str]) -> bool:
    return episode.sub_label in sub_labels


class SingleMovementAgent(Agent):
    def __init__(self, action: str = "right") -> None:
        assert action in FindViewActions.all
        self.action = action

    def reset(self):
        ...

    def act(self, observations):
        return self.action


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
    print(">>> Config:")
    print(cfg.pretty_text)

    # params
    split = 'test'
    sub_labels = ["restaurant"]

    # setup filter func
    filter_by_labels = partial(filter_episodes_by_sub_labels, sub_labels=sub_labels)

    # initialize dataset
    dataset = make_dataset(cfg=cfg, split=split, filter_fn=filter_by_labels)
    print(f"Using {len(dataset)}")  # 200
    episode_iterator = dataset.get_iterator(
        cycle=True,
        shuffle=False,
        num_episode_sample=30,
    )

    # initialze agent
    if args.agent == "single":
        agent = SingleMovementAgent(action="right")
    elif args.agent == "greedy":
        agent = GreedyMovementAgent(cfg=cfg, chance=0.001, seed=0)
    elif args.agent == "fm":
        agent = FeatureMatchingAgent(cfg=cfg)

    # initialize sim
    sim = FindViewSim(
        **cfg.sim,
    )
    sim.inititialize_loader(
        is_torch=True,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )
    # sim.inititialize_loader(
    #     is_torch=True,
    #     dtype=torch.float32,
    #     device=torch.device('cuda:0'),
    # )
    # sim.inititialize_loader(
    #     is_torch=False,
    #     dtype=np.float32,
    # )

    rot_tracker = RotationTracker(
        inc=cfg.step_size,
        pitch_threshold=cfg.pitch_threshold,
    )

    # run loops for testing
    num_iter = 2
    num_steps = 1000
    for i in range(num_iter):
        episode = next(episode_iterator)
        print(episode.img_name)

        initial_rotation = episode.initial_rotation
        target_rotation = episode.target_rotation

        print("init/target rot:", initial_rotation, target_rotation)

        rot_tracker.initialize(initial_rotation)
        # reset/init sim from episode
        pers, target = sim.reset(
            equi_path=episode.path,
            initial_rotation=initial_rotation,
            target_rotation=target_rotation,
        )
        obs = {'pers': pers, 'target': target}
        agent.reset()

        pers_list = []

        for j in tqdm(range(num_steps)):
            action = agent.act(obs)
            rot = rot_tracker.convert(action)
            pers = sim.move(rot)
            target = sim.target

            obs = {'pers': pers, 'target': target}

            render_pers = sim.render_pers()
            pers_list.append(render_pers)

        # save as video
        save_path = os.path.join('./results/sim', f"{args.agent}_{episode.img_name}.mp4")
        save_images_as_video(pers_list, save_path)

        # for rot in rot_tracker.history:
        #     print(rot)
