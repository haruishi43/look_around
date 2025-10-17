#!/usr/bin/env python3

import abc

from torch import nn as nn

from findview_baselines.rl.models.nets.base_net import Net
from ..utils import CategoricalNet, GaussianNet


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module, metaclass=abc.ABCMeta):
    action_distribution: nn.Module
    net: Net

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

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

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

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass
