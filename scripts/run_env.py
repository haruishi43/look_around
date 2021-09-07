#!/usr/bin/env python3

"""Run `Env`

Before building parallel environments, we need to test the basic environments



"""

import argparse
import json
import os

from mycv.utils import Config
import torch
from tqdm import tqdm

from LookAround.FindView.env import FindViewActions, make_env
from LookAround.utils.visualizations import save_images_as_video


class SingleMovementAgent(object):
    def __init__(self, action: str = "right") -> None:
        assert action in FindViewActions.all
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
    cfg.max_steps = 2000
    print(">>> Config:")
    print(cfg.pretty_text)

    # params:
    split = 'train'
    is_torch = True
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 5000

    # initialize env
    env = make_env(
        cfg=cfg,
        split=split,
        is_torch=is_torch,
        device=device,
    )
    # initialize agent
    agent = SingleMovementAgent(action="right")

    images = []
    obs = env.reset()
    print(obs.keys())
    render = env.render()
    images.append(render['target'])
    images.append(render['pers'])

    for i in tqdm(range(num_steps)):
        action = agent.act()
        obs = env.step(action)
        pers = env.render()['pers']
        images.append(pers)
        if env.episode_over:
            print("next episode!")
            # save stats to file
            stats = env.get_metrics()
            img_name = env.current_episode.img_name
            save_path = os.path.join('./results', f'{img_name}.json')
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=2)

            env.reset()

    save_path = os.path.join('./results', 'test_env.mp4')
    save_images_as_video(images, save_path)
