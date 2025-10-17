#!/usr/bin/env python3

import torch
from torch import Size, Tensor
from torch import nn as nn


class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:  # noqa: B008
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:  # noqa: B008
        return super().rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        ret = super().log_prob(actions).sum(-1).unsqueeze(-1)
        return ret


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        action_activation="tanh",
        use_log_std=False,
        use_softplus=False,
        min_log_std=1e-6,
        max_log_std=1,
        min_std=-5,
        max_std=2,
    ) -> None:
        super().__init__()

        self.action_activation = action_activation
        self.use_log_std = use_log_std
        self.use_softplus = use_softplus
        if self.use_log_std:
            self.min_std = min_log_std
            self.max_std = max_log_std
        else:
            self.min_std = min_std
            self.max_std = max_std

        self.mu = nn.Linear(num_inputs, num_outputs)
        self.std = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.orthogonal_(self.std.weight, gain=0.01)
        nn.init.constant_(self.std.bias, 0)

    def forward(self, x: Tensor) -> CustomNormal:
        mu = self.mu(x)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        std = torch.clamp(self.std(x), min=self.min_std, max=self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std)
