#!/usr/bin/env python3

from typing import Optional

from gymnasium import spaces
from mmengine import Config

from .base_policy import Policy
from ..nets import build_net


class BasicPolicy(Policy):
    def __init__(
        self,
        net,
        action_space,
        pretrained_path: Optional[str] = None,
        pretrained_prefix: str = "actor_critic.",
        **kwargs,
    ):
        super().__init__(
            net,
            action_space.n,
            **kwargs,
        )

        # pretrained weights variables
        self.can_load_pretrained_weights = True if pretrained_path else False
        self.pretrained_path = pretrained_path
        self.pretrained_prefix = pretrained_prefix

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # initialize network
        net = build_net(
            config=config.net,
            observation_space=observation_space,
            action_space=action_space,
        )

        return cls(
            net=net,
            action_space=action_space,
            **kwargs,
        )
