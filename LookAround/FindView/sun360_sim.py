#!/usr/bin/env python3

from typing import Dict, Union

from addict import Dict as ADDict

from mycv.image import imread

from .equi2pers import Equi2Pers


def load_image(img_path: str):
    img = imread(img_path)
    # - Convert to torch?
    # - Read as PIL and add image augmentation?
    return img


class View:
    _yaw: int
    _pitch: int

    def __init__(self, yaw, pitch) -> None:
        self._yaw = yaw
        self._pitch = pitch

    def to_dict(
        self,
        to_rad: bool = True,
    ) -> Dict[str, Union[int, float]]:
        _yaw = self._yaw  # TODO: convert to radian?
        _pitch = self._pitch

        return dict(
            roll=0.0,  # View is set to roll=0.0
            pitch=_pitch,
            yaw=_yaw,
        )


class SUN360Data:
    img_path: str
    initial_view: View
    target_view: View


# NOTE: Single Env (NOT PARALLEL)
class SUN360Sim:

    def __init__(
        self,
        config: ADDict,
    ) -> None:
        assert isinstance(config, ADDict), \
            f"ERR: {type(config)} is not a valid config"

        self.equi2pers = Equi2Pers(**config.equi2pers)
        self.equi = None

    def load(
        self,
        img_path: str,
    ) -> None:
        """Loads new scene for the simulator
        """
        self.equi = load_image(img_path=img_path)

    def look_at(self, rot):

        assert self.equi is not None, \
            "ERR: self.equi not set"
        img = self.equi2pers(rot, self.equi)
        return img

    def reset(
        self,
    ) -> None:
        self.equi = None

    def close(self) -> None:
        self.equi = None
        self.equi2pers = None
