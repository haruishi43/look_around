#!/usr/bin/env python3

from typing import Dict, List

from LookAround.FindView.actions import FindViewActions


class RotationTracker(object):

    rot: Dict[str, int]
    history: List[Dict[str, int]]

    def __init__(
        self,
        inc: int = 1,
        pitch_threshold: int = 60,
    ) -> None:
        self.inc = inc
        self.pitch_threshold = pitch_threshold

        self.rot = None
        self.history = []

    def initialize(self, initial_rotation: Dict[str, int]) -> None:
        self.rot = initial_rotation
        self.history = []

    def convert(self, action: str) -> Dict[str, float]:
        assert self.rot is not None
        pitch = self.rot['pitch']
        yaw = self.rot['yaw']

        if action == FindViewActions.UP:
            pitch += self.inc

        elif action == FindViewActions.DOWN:
            pitch -= self.inc

        elif action == FindViewActions.RIGHT:
            yaw += self.inc

        elif action == FindViewActions.LEFT:
            yaw -= self.inc

        if pitch >= self.pitch_threshold:
            pitch = self.pitch_threshold
        elif pitch <= -self.pitch_threshold:
            pitch = -self.pitch_threshold
        if yaw > 180:
            yaw -= 2 * 180
        elif yaw <= -180:
            yaw += 2 * 180

        rot = {
            "roll": 0,
            "pitch": int(pitch),
            "yaw": int(yaw),
        }

        self.rot = rot
        self.history.append(rot)

        return rot
