#!/usr/bin/env python3

import math

from typing import (
    List,
    Dict,
)

import numpy as np


class Rot:
    _yaw: int
    _pitch: int
    _roll: int

    def __init__(self, pitch_threshold: int):
        assert pitch_threshold <= 90 and pitch_threshold > 0, "ERR: add correct pitch threshold"
        self._pitch_threshold = pitch_threshold
        self.pi: float = math.pi
        self.zero()

    def zero(self):
        self._yaw = 0
        self._pitch = 0
        self._roll = 0

    def _check_range_yaw(self):
        if self._yaw >= 180:
            self._yaw = self._yaw - 2 * 180
        elif self._yaw <= -180:
            self._yaw = self._yaw + 2 * 180

    def _check_range_pitch(self):
        if self._pitch >= self._pitch_threshold:
            self._pitch = self._pitch_threshold
        elif self._pitch <= - self._pitch_threshold:
            self._pitch = - self._pitch_threshold

    def _check_range_roll(self):
        if self._roll >= 180:
            self._roll = self._roll - 2 * 180
        elif self._roll <= -180:
            self._roll = self._roll + 2 * 180

    def _check_all(self):
        self._check_range_yaw()
        self._check_range_pitch()

    def inc_yaw(self, inc: int) -> None:
        self._yaw = self._yaw + inc
        self._check_range_yaw()

    def inc_pitch(self, inc: int) -> None:
        self._pitch = self._pitch + inc
        self._check_range_pitch()

    @property
    def yaw(self) -> int:
        return self._yaw

    @property
    def pitch(self) -> int:
        return self._pitch

    @property
    def roll(self) -> int:
        return self._roll

    @yaw.setter
    def yaw(self, deg: int) -> None:
        self._yaw = deg
        self._check_range_yaw()

    @pitch.setter
    def pitch(self, deg: int) -> None:
        self._pitch = deg
        self._check_range_pitch()

    @roll.setter
    def roll(self, deg: int) -> None:
        self._roll = deg
        self._check_range_roll()

    @property
    def rad(self) -> List[float]:
        return [
            self._yaw * self.pi / 180,
            self._pitch * self.pi / 180,
            self._roll * self.pi / 180
        ]

    @property
    def deg(self) -> List[int]:
        return [self._yaw, self._pitch, self._roll]

    @rad.setter
    def rad(self, rad: List[float]) -> None:
        """Assume [yaw, pitch, roll]
        """
        self._yaw = int(round(rad[0] * 180 / self.pi))
        self._pitch = int(round(rad[1] * 180 / self.pi))
        self._roll = int(round(rad[2] * 180 / self.pi))
        self._check_all()

    @deg.setter
    def deg(self, deg: List[int]) -> None:
        """Assume [yaw, pitch, roll]
        """
        self._yaw = deg[0]
        self._pitch = deg[1]
        self._roll = deg[2]
        self._check_all()


class RotationTracker:
    """Keep track of rotation
    """

    def __init__(
        self,
        inc: int = 1,
        pitch_thresh: int = 60,
    ) -> None:
        self.inc: int = inc
        self.rot = Rot(pitch_thresh)

    def update(self, action: int, rad: bool = True):
        """Update rotation based on action
        Returns:
            rotations as List
        """
        if action == SimulatorActions.UP:
            self.rot.inc_pitch(- self.inc)
        elif action == SimulatorActions.DOWN:
            self.rot.inc_pitch(self.inc)
        elif action == SimulatorActions.RIGHT:
            self.rot.inc_yaw(self.inc)
        elif action == SimulatorActions.LEFT:
            self.rot.inc_yaw(- self.inc)

        # ADD Action here:
        # TODO: make changing action easier

        if rad:
            return self.rot.rad
        else:
            return self.rot.deg

    @property
    def rad(self) -> List[float]:
        return self.rot.rad

    @property
    def deg(self) -> List[int]:
        return self.rot.deg

    @deg.setter
    def deg(self, rot: List[int]) -> None:
        """set rotation as degrees
        """
        self.rot.deg = rot

    @rad.setter
    def rad(self, rot: List[float]) -> None:
        self.rot.deg = rot

    def reset(self):
        """Reset the rotations to initial position
        """
        self.rot.zero()

    def __repr__(self):
        return "<Object:{0} Rotation: [yaw: {1}, pitch: {2}, roll: {3}]>".format(
            id(self), self.rot.yaw, self.rot.pitch, self.rot.roll)
