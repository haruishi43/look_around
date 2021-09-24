#!/usr/bin/env python3

import torch

from LookAround.config import Config
from LookAround.FindView.env import FindViewEnv


def test_env():
    cfg = Config.fromfile("tests/configs/test_env.py")
    print(cfg.pretty_text)

    env = FindViewEnv.from_config(
        cfg=cfg,
        split="test",
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    assert len(env.episodes) == env.number_of_episodes

    action = "stop"
    for i in range(env.number_of_episodes):
        _ = env.reset()

        while not env.episode_over:
            _ = env.step(action)


def test_env_2():
    cfg = Config.fromfile("tests/configs/test_env.py")
    print(cfg.pretty_text)

    # check for termination
    cfg.dataset.max_steps = 10

    env = FindViewEnv.from_config(
        cfg=cfg,
        split="test",
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    assert len(env.episodes) == env.number_of_episodes

    action = "right"
    for i in range(env.number_of_episodes):
        _ = env.reset()

        while not env.episode_over:
            _ = env.step(action)


def test_env_3():
    cfg = Config.fromfile("tests/configs/test_env.py")
    print(cfg.pretty_text)

    env = FindViewEnv.from_config(
        cfg=cfg,
        split="train",
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    num_episodes = 10

    action = "stop"
    for i in range(num_episodes):
        _ = env.reset()

        episode = env.current_episode
        print(episode.difficulty)

        while not env.episode_over:
            _ = env.step(action)

    # change difficulty
    env.change_difficulty('hard')

    action = "stop"
    for i in range(num_episodes):
        _ = env.reset()

        episode = env.current_episode
        print(episode.difficulty)

        while not env.episode_over:
            _ = env.step(action)
