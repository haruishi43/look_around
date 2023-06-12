#!/usr/bin/env python3

from os import PathLike
from typing import Dict

import attr

from LookAround.utils.utils import not_none_validator


@attr.s(auto_attribs=True, kw_only=True)
class Episode:
    episode_id: int = attr.ib(default=None, validator=not_none_validator)
    img_name: str = attr.ib(default=None, validator=not_none_validator)
    path: PathLike = attr.ib(default=None, validator=not_none_validator)
    label: str = attr.ib(default=None, validator=not_none_validator)
    sub_label: str = attr.ib(default=None, validator=not_none_validator)
    initial_rotation: Dict[str, int] = attr.ib(
        default=None, validator=not_none_validator
    )
    target_rotation: Dict[str, int] = attr.ib(
        default=None, validator=not_none_validator
    )
    difficulty: str = attr.ib(default=None, validator=not_none_validator)
    steps_for_shortest_path: int = attr.ib(
        default=None, validator=not_none_validator
    )


@attr.s(auto_attribs=True, kw_only=True)
class PseudoEpisode:
    img_name: str = attr.ib(default=None, validator=not_none_validator)
    path: PathLike = attr.ib(default=None, validator=not_none_validator)
    label: str = attr.ib(default=None, validator=not_none_validator)
    sub_label: str = attr.ib(default=None, validator=not_none_validator)
