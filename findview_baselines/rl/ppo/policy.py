#!/usr/bin/env python3

import abc

import torch
from gymnasium import spaces
from torch import nn as nn

from findview_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from findview_baselines.rl.models.simple_cnn import SimpleCNN
from findview_baselines.utils.common import CategoricalNet, GaussianNet


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        net,
        dim_actions,
        action_distribution_type="categorical",
        use_log_std=False,
        use_softplus=False,
        min_std=1e-6,
        max_std=1,
        min_log_std=-5,
        max_log_std=2,
        action_activation="tanh",
    ) -> None:
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        if action_distribution_type is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = action_distribution_type

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                action_activation=action_activation,
                use_log_std=use_log_std,
                use_softplus=use_softplus,
                min_std=min_std,
                max_std=max_std,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class FindViewBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        **kwargs,
    ):
        super().__init__(
            FindViewBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
            ),
            action_space.n,
            **kwargs,
        )


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


class FindViewBaselineNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
    ):
        super().__init__()

        goal_observation_space = spaces.Dict(
            {"target": observation_space.spaces["target"]}
        )
        self.goal_visual_encoder = SimpleCNN(
            goal_observation_space, hidden_size
        )
        self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.goal_visual_encoder({"target": observations["target"]})

        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        return x_out, rnn_hidden_states
