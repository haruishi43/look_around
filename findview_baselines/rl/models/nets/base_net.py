#!/usr/bin/env python3

import abc

import torch.nn as nn
from gymnasium import spaces

from mmengine import Config, Registry

NetRegistry = Registry('net')


def build_net(
    cfg: Config,
    observation_space: spaces.Dict,
    action_space: spaces.Dict,
    **kwargs,
):
    kwargs.update(
        dict(
            observation_space=observation_space,
            action_space=action_space,
        )
    )
    return NetRegistry.build(cfg, **kwargs)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass
