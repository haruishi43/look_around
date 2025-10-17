#!/usr/bin/env python3

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from gymnasium import spaces

from mmengine import Config, Registry

EncoderRegistry = Registry(name="encoder")

VISUAL_SENSORS_UUID = [
    "rgb",
    # "depth",
    # "semantic",
]


def build_encoder(
    cfg: Config,
    observation_space: spaces.Dict,
    **kwargs,
):
    """Build encoder.

    This function is used since we want to update some arguments that are not
    available from the config.
    """
    kwargs.update(dict(observation_space=observation_space))
    return EncoderRegistry.build(cfg, **kwargs)


class BaseEncoder(nn.Module):

    _n_input_dict: dict
    _output_shape: Optional[Union[tuple, list]]
    _normalize: Optional[Union[nn.Module, object]]

    def __init__(
        self,
        observation_space: spaces.Dict,
        normalize_inputs: bool = True,
        use_running_mean_var: bool = False,
    ):
        super().__init__()

        # initialize the number of inputs for each sensor
        self._n_input_dict = self._init_observation(observation_space)
        assert self._n_input_dict["rgb"] > 0, "rgb input is required"

        # normalize inputs
        if normalize_inputs or use_running_mean_var:
            from .normalization import RunningMeanAndVar

            self._normalize = RunningMeanAndVar(self.in_channels)
        elif normalize_inputs:
            from .normalization import StaticMeanVar

            # NOTE: we just use the default mean and var from torchvision
            self._normalize = StaticMeanVar(
                mean=(0.485, 0.456, 0.406),
                var=(0.229, 0.224, 0.225),
            )
        else:
            self._normalize = None

    @staticmethod
    def _init_observation(observation_space: spaces.Dict):
        n_input_dict = {k: 0 for k in VISUAL_SENSORS_UUID}
        for k, v in observation_space.spaces.items():
            if k in n_input_dict:
                n_input_dict[k] = v.shape[2]
            else:
                n_input_dict[k] = 0
        return n_input_dict

    @property
    def in_channels(self):
        n_inputs = 0
        for v in self._n_input_dict.values():
            n_inputs += v
        return n_inputs

    @property
    def output_shape(self):
        assert self._output_shape is not None, "output shape is not initialized"
        return self._output_shape

    @property
    def is_blind(self):
        n_inputs = 0
        for v in self._n_input_dict.values():
            n_inputs += v
        return n_inputs == 0

    def _normalize_inputs(self, x: Dict[str, torch.Tensor]):
        """Normalize inputs."""
        if self._normalize is not None:
            x = self._normalize(x)
        return x

    def layer_init(self):
        pass

    def _forward(self, x):
        """Forward network."""
        raise NotImplementedError

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if self.is_blind:
            return None
        else:
            # NOTE: different from habtiat, the input is already standarized
            # between 0~1 in CHW format.

            # concat to single tensor
            x = torch.cat([v for v in inputs.values()], dim=1)

            # normalize inputs
            with torch.no_grad():
                # we don't want gradients to interfere with normalization
                x = self._normalize_inputs(x)

            # forward model
            return self._forward(x)
