#!/usr/bin/env python3

"""
Control agent training and evaluation

- Communication with simulators
- Observations (obs)
- Rewards
- Action
- Dataset
- Episode Scheduler

- Vectorized Env uses this

"""

# NOTE: how can we generalize the actions so that it can be interpreted in the simulator?
# is it even the simulator's job to interpret actions? only process raw rotations?
# Maybe in the `Env`, we can add functionality to interpret actions and keep track of
# rotations


class Action(object):
    ...


class UpAction(Action):
    ...


class DownAction(Action):
    ...


class RightAction(Action):
    ...


class LeftAction(Action):
    ...


class StopAction(Action):
    ...


ActionSpace = {
    "up": UpAction,
    "down": DownAction,
    "right": RightAction,
    "left": LeftAction,
    "stop": StopAction,
}


class FindViewEnv(object):

    def __init__(
        self,
        config,
        dataset,  # Optional -> call `make_dataset` using `config`
    ) -> None:
        ...

    def step(self, action: Action):
        ...

    def reset(self):
        ...

    def render(self):
        ...


class FindViewRLEnv(object):

    def __init__(
        self,
        cfg:
    ) -> None:
        ...
