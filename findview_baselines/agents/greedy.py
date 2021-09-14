#!/usr/bin/env python3

import random

from LookAround.core.agent import Agent
from LookAround.FindView.actions import FindViewActions


def movement_generator(size=4):
    """`size` is number of actions
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
    def __init__(self, chance=0.001, seed=0) -> None:
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
        # FIXME: change stop criteria
        if self.rs.random() < self.stop_chance:
            return self.stop_action
        return self.movement_actions[next(self.g)]
