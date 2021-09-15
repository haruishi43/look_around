#!/usr/bin/env python3

"""
Agent that greedly moves around from the initial rotation
and stops when the target image is found
"""

import random

import numpy as np
import torch

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.FindView.actions import FindViewActions


def movement_generator(size=4):
    """`size` is number of actions
    This movement generator moves around the initial point
    """
    idx = 0
    repeat = 1
    while True:
        for r in range(repeat):
            yield idx

        idx = (idx + 1) % size
        if idx % 2 == 0:
            repeat += 1


class GreedyMovementAgent(Agent):
    def __init__(
        self,
        cfg: Config,
        chance: float = 0.001,
        seed: int = 0,
    ) -> None:
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        self.stop_chance = chance
        for action in self.movement_actions:
            assert action in FindViewActions.all
        self.g = movement_generator(len(self.movement_actions))
        self.rs = random.Random(seed)

    def reset(self):
        self.g = movement_generator(len(self.movement_actions))

    def act(self, observations):

        pers = observations['pers']
        target = observations['target']

        close = False
        if torch.is_tensor(pers):
            close = torch.allclose(pers, target)
        elif isinstance(pers, np.ndarray):
            close = np.allclose(pers, target)
        else:
            # NOTE: using random change to stop
            if self.rs.random() < self.stop_chance:
                close = True

        if close:
            return self.stop_action

        return self.movement_actions[next(self.g)]


def main():

    import argparse

    from LookAround.core.logging import logger
    from LookAround.FindView.benchmark import FindViewBenchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    agent = GreedyMovementAgent(cfg)
    benchmark = FindViewBenchmark(
        cfg=cfg,
        device=torch.device('cpu'),
    )
    metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)

    for k, v in metrics.items():
        logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
