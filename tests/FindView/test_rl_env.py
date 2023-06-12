#!/usr/bin/env python3

import torch

from LookAround.config import Config
from LookAround.FindView.rl_env import FindViewRLEnv, RLEnvRegistry


def test_rlenv():
    cfg = Config.fromfile("tests/configs/rl_env_1.py")
    print(cfg.pretty_text)

    num_steps = 4000

    rlenv: FindViewRLEnv = RLEnvRegistry.build(
        cfg,
        name=cfg.rl_env.name,
        split="train",
        filter_fn=None,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    obs = rlenv.reset()

    for i in range(num_steps):
        action = "right"
        obs, reward, done, info = rlenv.step(action)
        # print("reward", reward)
        if done:
            if action == "stop":
                print("called stop")
                assert info['called_stop']
            print(">>> next episode")
            obs = rlenv.reset()
