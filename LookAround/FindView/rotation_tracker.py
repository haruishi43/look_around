#!/usr/bin/env python3

"""Rotation Tracker for the simulator

This class can be subclassed and overwritten to change the agents actions

Current implementation is an incremental movement tracker
"""

from copy import deepcopy
from typing import Dict, List

from LookAround.FindView.actions import FindViewActions


class RotationTracker(object):

    _rot: Dict[str, int]
    _history: List[Dict[str, int]]

    def __init__(
        self,
        inc: int = 1,
        pitch_threshold: int = 60,
    ) -> None:
        """Rotation Tracker

        params:
        - inc (int): default 1
        - pitch_threshold (int): default 60
        """
        assert isinstance(inc, int)
        assert isinstance(pitch_threshold, int)
        assert 0 < inc <= 90
        assert 30 <= pitch_threshold <= 90

        self.inc = inc
        self.pitch_threshold = pitch_threshold

        # initialize as empty
        self._rot = None
        self._history = []

    @property
    def history(self) -> List[Dict[str, int]]:
        assert len(self._history) > 0
        return deepcopy(self._history)

    @property
    def current_rotation(self) -> Dict[str, int]:
        assert self._rot is not None
        return deepcopy(self._rot)

    def reset(self, initial_rotation: Dict[str, int]) -> None:
        """Reset the tracker; initialize rotation and history
        """
        self._rot = initial_rotation
        self._history = [initial_rotation]

    def move(self, action: str) -> Dict[str, int]:
        """Alias for `action2rot`
        """
        return self.action2rot(action)

    def action2rot(self, action: str) -> Dict[str, int]:
        """Convert `action` to simulator friendly `rotation`
        """
        assert self._rot is not None
        pitch = self._rot['pitch']
        yaw = self._rot['yaw']

        # shift according to input action
        if action == FindViewActions.UP:
            pitch += self.inc
        elif action == FindViewActions.DOWN:
            pitch -= self.inc
        elif action == FindViewActions.RIGHT:
            yaw += self.inc
        elif action == FindViewActions.LEFT:
            yaw -= self.inc
        else:
            raise ValueError(f"Invalid action called: {action}")

        # upper and lower bounds for pitch
        if pitch >= self.pitch_threshold:
            pitch = self.pitch_threshold
        elif pitch <= -self.pitch_threshold:
            pitch = -self.pitch_threshold

        # wrap around for yaw
        if yaw > 180:
            yaw -= 2 * 180
        elif yaw <= -180:
            yaw += 2 * 180

        # current rotation
        rot = {
            "roll": 0,
            "pitch": int(pitch),
            "yaw": int(yaw),
        }

        # keep track of moving rotation
        self._rot = rot
        self._history.append(rot)

        return rot
