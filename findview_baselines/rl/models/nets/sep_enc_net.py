#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from mmengine import Config

from findview_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from findview_baselines.rl.models.encoders import BaseEncoder, build_encoder
from .base_net import Net, NetRegistry
from .modules import PrevActionEmbedding


@NetRegistry.register_module()
class SepEncNet(Net):
    """Separated Encoders for Goal and Obs."""

    visual_encoder: BaseEncoder
    goal_encoder: BaseEncoder
    can_load_pretrained_weights: bool = False

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Dict,
        encoder_config: Config,
        goal_encoder_config: Config,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        embedding_size: int = 256,
        goal_embedding_size: int = 256,
        pretrained_path: Optional[str] = None,
        pretrained_prefix: str = "actor_critic.net.",
        **kwargs,
    ):
        super().__init__()

        # pretrained weights variables
        self.can_load_pretrained_weights = True if pretrained_path else False
        self.pretrained_path = pretrained_path
        self.pretrained_prefix = pretrained_prefix

        # setup variables
        self._hidden_size = hidden_size

        goal_observation_space = spaces.Dict(
            {"rgb": observation_space.spaces["target"]}
        )
        self.goal_encoder = build_encoder(
            cfg=goal_encoder_config,
            observation_space=goal_observation_space,
            **kwargs,
        )
        self._n_input_goal = hidden_size

        vis_observation_space = spaces.Dict(
            {"rgb": observation_space.spaces["pers"]}
        )
        self.visual_encoder = build_encoder(
            cfg=encoder_config,
            observation_space=vis_observation_space,
            **kwargs,
        )

        # initialize action embedding
        self.prev_action_embedding = PrevActionEmbedding(
            action_space=action_space,
            embedding_size=32,
        )
        rnn_input_size = self.prev_action_embedding.output_size

        # add visual embedding layer
        if not self.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape),
                    embedding_size,
                ),
                nn.ReLU(True),
            )
            rnn_input_size += embedding_size

        # add goal embedding layer
        self.goal_visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.goal_encoder.output_shape),
                goal_embedding_size,
            ),
            nn.ReLU(True),
        )
        rnn_input_size += goal_embedding_size

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            visual_feats = observations.get(
                "visual_feats",
                self.visual_encoder({'rgb': observations["pers"]}),
            )
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        target_feats = self.goal_encoder(
            {"rgb": observations["target"]}
        )
        target_feats = self.goal_visual_fc(target_feats)
        x.append(target_feats)

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
        )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states

    def load_pretrained_weights(self):
        # initialize pretrained weights in childrens
        self.visual_encoder.load_pretrained_weights()

        if self.can_load_pretrained_weights:
            # load pretrained weights
            pretrained_state = torch.load(self.pretrained_path, map_location="cpu")

            # NOTE: we can initialize using the entire state_dict, but I decided to
            # initialize each layer separately to make it easier to debug

            # visual encoder
            visual_encoder_prefix = self.pretrained_prefix + "visual_encoder."
            self.visual_encoder.load_state_dict(
                {
                    k[len(visual_encoder_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(visual_encoder_prefix)
                }
            )

            # visual fc
            visual_fc_prefix = self.pretrained_prefix + "visual_fc."
            self.visual_fc.load_state_dict(
                {
                    k[len(visual_fc_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(visual_fc_prefix)
                }
            )

            # goal encoder
            goal_encoder_prefix = self.pretrained_prefix + "goal_encoder."
            self.goal_encoder.load_state_dict(
                {
                    k[len(goal_encoder_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(goal_encoder_prefix)
                }
            )

            # goal visual fc
            goal_visual_fc_prefix = self.pretrained_prefix + "goal_visual_fc."
            self.goal_visual_fc.load_state_dict(
                {
                    k[len(goal_visual_fc_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(goal_visual_fc_prefix)
                }
            )

            # action embedding
            action_embedding_prefix = self.pretrained_prefix + "prev_action_embedding."
            self.prev_action_embedding.load_state_dict(
                {
                    k[len(action_embedding_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(action_embedding_prefix)
                }
            )

            # rnn
            rnn_prefix = self.pretrained_prefix + "state_encoder."
            self.state_encoder.load_state_dict(
                {
                    k[len(rnn_prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(rnn_prefix)
                }
            )

    def freeze_visual_encoder(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)
