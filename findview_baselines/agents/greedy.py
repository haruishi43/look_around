#!/usr/bin/env python3

"""
Agent that greedly moves around from the initial rotation
and stops when the target image is found

FIXME:
- [ ] is the movement generator really greedy?
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
        chance: float = 0.001,
        seed: int = 0,
    ) -> None:
        self.name = "greedy"
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        self.stop_chance = chance

        for action in self.movement_actions:
            assert action in FindViewActions.all

        self.g = movement_generator(len(self.movement_actions))
        self.rs = random.Random(seed)

    @classmethod
    def from_config(cls, cfg: Config):
        agent_cfg = cfg.greedy
        return cls(
            chance=agent_cfg.chance,
            seed=agent_cfg.seed,
        )

    def reset(self):
        self.g = movement_generator(len(self.movement_actions))

    def act(self, observations):
        pers = observations["pers"]
        target = observations["target"]

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

    from LookAround.config import DictAction
    from LookAround.FindView.benchmark import FindViewBenchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the agent (used for naming save directory)",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="arguments in dict",
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    print(">>> Config:")
    print(cfg.pretty_text)

    # Intializing the agent
    agent = GreedyMovementAgent.from_config(cfg)
    name = agent.name
    if args.name is not None:
        name += "_" + args.name

    # Benchmark
    print(f"Benchmarking {name}")
    benchmark = FindViewBenchmark(
        cfg=cfg,
        agent_name=name,
    )
    benchmark.evaluate(agent)


if __name__ == "__main__":
    main()
