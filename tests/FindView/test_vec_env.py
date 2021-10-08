#!/usr/bin/env python3

"""
TODO:
- [x] Test Threaded
- [ ] Test Multiprocessing
- [ ] Test EquilibVecEnv
"""

import random

# import numpy as np
import torch
from tqdm import tqdm

import pytest

from LookAround.config import Config
from LookAround.FindView.env import FindViewActions
from LookAround.FindView.vec_env import construct_envs


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
    def __init__(self, chance: float = 0.01, seed: int = 0) -> None:
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        self.stop_chance = chance
        for action in self.movement_actions:
            assert action in FindViewActions.all
        self.g = movement_generator(len(self.movement_actions))

        self.rst = random.Random(seed)

    def act(self):
        if self.rst.random() < self.stop_chance:
            return self.stop_action
        return self.movement_actions[next(self.g)]

    def reset(self):
        self.g = movement_generator(len(self.movement_actions))


@pytest.mark.skip(reason="skipping for now")
def test_vec_env():
    cfg = Config.fromfile("tests/configs/vec_env_1.py")
    print(cfg.pretty_text)

    # params:
    vec_type = "threaded"
    split = 'train'
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 500

    envs = construct_envs(
        cfg=cfg,
        split=split,
        is_rlenv=False,
        dtype=dtype,
        device=device,
        vec_type=vec_type,
    )

    agents = [GreedyMovementAgent() for _ in range(cfg.base_trainer.num_envs)]

    assert envs.num_envs == cfg.base_trainer.num_envs

    # reset env
    _ = envs.reset()

    for _ in tqdm(range(num_steps)):

        actions = [agent.act() for agent in agents]

        _ = envs.step(actions)
        dones = envs.episode_over()

        for i, done in enumerate(dones):
            if done:
                print(f"Loading next episode for {i}")
                # NOTE: Env needs to be reset
                envs.reset_at(i)
                agents[i].reset()


@pytest.mark.skip(reason="skipping for now")
def test_rl_vec_env():
    cfg = Config.fromfile("tests/configs/vec_rl_env_1.py")
    print(cfg.pretty_text)

    # params:
    vec_type = "threaded"
    split = 'train'
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 500

    envs = construct_envs(
        cfg=cfg,
        split=split,
        is_rlenv=True,
        dtype=dtype,
        device=device,
        vec_type=vec_type,
    )

    agents = [GreedyMovementAgent(seed=cfg.seed) for _ in range(cfg.base_trainer.num_envs)]

    assert envs.num_envs == cfg.base_trainer.num_envs

    # reset env
    _ = envs.reset()

    for _ in tqdm(range(num_steps)):

        actions = [agent.act() for agent in agents]

        outputs = envs.step(actions)
        _, _, dones, infos = [list(x) for x in zip(*outputs)]

        for i, done in enumerate(dones):
            if done:
                if actions[i] == "stop":
                    print(f"{i} called stop")
                    assert infos[i]['called_stop']

                print(f"Loading next episode for {i}")
                agents[i].reset()


def test_rl_vec_env_reproducibility():
    cfg = Config.fromfile("tests/configs/vec_rl_env_1.py")
    print(cfg.pretty_text)

    # params:
    vec_type = "threaded"
    split = 'train'
    dtype = torch.float32
    device = torch.device('cpu')
    num_steps = 500

    envs = construct_envs(
        cfg=cfg,
        split=split,
        is_rlenv=True,
        dtype=dtype,
        device=device,
        vec_type=vec_type,
    )

    agents = [GreedyMovementAgent(seed=cfg.seed) for _ in range(cfg.base_trainer.num_envs)]

    assert envs.num_envs == cfg.base_trainer.num_envs

    # reset env
    _ = envs.reset()

    episodes = envs.current_episodes()
    for i, episode in enumerate(episodes):
        print(i, episode.img_name)
        print("\t", episode.initial_rotation, episode.target_rotation)

    for _ in tqdm(range(num_steps)):

        actions = [agent.act() for agent in agents]

        outputs = envs.step(actions)
        _, _, dones, infos = [list(x) for x in zip(*outputs)]

        for i, done in enumerate(dones):
            if done:
                if actions[i] == "stop":
                    print(f"{i} called stop")
                    assert infos[i]['called_stop']

                print(f"Loading next episode for {i}")
                agents[i].reset()

                episode = envs.current_episodes()[i]
                print(i, episode.img_name)
                print("\t", episode.initial_rotation, episode.target_rotation)