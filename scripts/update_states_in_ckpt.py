#!/usr/bin/env python3

"""Script to update old checkpoint states to new ones for continued training
"""

import argparse
import os

import torch

from LookAround.config import Config

from findview_baselines.utils.common import get_last_checkpoint_folder

from findview_baselines.rl.ppo.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Print the whole config")
    parser.add_argument("config", help="config file path")
    args = parser.parse_args()
    return args


# NOTE: edit this before updating
new_extra_state = dict(
    env_time=81940.986,
    pth_time=82041.861,
    count_checkpoints=20,
    num_steps_done=20480000,
    num_updates_done=10000,
    _last_checkpoint_percent=1.0,
    prev_time=169271.85478535,
    running_episode_stats=dict(
        count=torch.zeros(16, 1),
        reward=torch.zeros(16, 1),
    ),
    window_episode_stats=None,
)


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print(f"Config:\n{cfg.pretty_text}")

    trainer = PPOTrainer(cfg)

    ckpt_dir = trainer.ckpt_dir
    ckpt_path = get_last_checkpoint_folder(ckpt_dir)
    d = trainer.load_checkpoint(ckpt_path, map_location="cpu")

    print(d.keys())
    print(d["extra_state"])

    print(os.path.basename(ckpt_path))

    trainer.save_checkpoint(
        os.path.basename(ckpt_path),
        dict(
            state_dict=d["state_dict"],
            cfg=d["cfg"],
        ),
        extra_state=new_extra_state,
    )
