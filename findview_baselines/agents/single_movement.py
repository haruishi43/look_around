#!/usr/bin/env python3

"""
FIXME: remove this agent since it doesn't do much; sole purpose is for debugging env
"""

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.FindView.actions import FindViewActions


class SingleMovementAgent(Agent):
    def __init__(self, action: str = "right") -> None:
        assert action in FindViewActions.all
        self.action = action
        self.name = "single"

    @classmethod
    def from_config(cls, cfg: Config):
        return cls(
            action=cfg.sm.action,
        )

    def act(self, observation):
        return self.action

    def reset(self):
        ...
