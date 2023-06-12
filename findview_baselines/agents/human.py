#!/usr/bin/env python3

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.FindView.actions import FindViewActions


class Human(Agent):
    def __init__(self):
        self.name = "human"

    @classmethod
    def from_config(cls, cfg: Config):
        # FIXME: does nothing; only for consistency
        return cls()

    def reset(self):
        ...

    def act(self, k):
        # NOTE: need to call cv2.waitKey(0) and get the return value

        if k == ord("w"):
            ret = "up"
        elif k == ord("s"):
            ret = "down"
        elif k == ord("a"):
            ret = "left"
        elif k == ord("d"):
            ret = "right"
        elif k == ord("q"):
            ret = "stop"
        else:
            raise ValueError(f"Pressed {k}")

        assert ret in FindViewActions.all
        return ret
