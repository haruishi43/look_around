#!/usr/bin/env python3

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import torch

from findview_baselines.common.tensor_dict import TensorDict


class RolloutStorage:
    """Class for storing rollout information for RL trainers."""

    _num_envs: int
    _num_steps: int

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_space,
        action_space: Optional[Any],
        recurrent_hidden_state_size: int,
        num_recurrent_layers: int = 1,
        action_shape: Optional[Tuple[int]] = None,
        discrete_actions: bool = True,
    ) -> None:

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        # initialize rollout buffer
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()
        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        num_steps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )
        self.buffers["recurrent_hidden_states"] = torch.zeros(
            (
                num_steps + 1,
                num_envs,
                num_recurrent_layers,
                recurrent_hidden_state_size,
            ),
        )
        self.buffers["rewards"] = torch.zeros(
            (num_steps + 1, num_envs, 1),
        )
        self.buffers["value_preds"] = torch.zeros(
            (num_steps + 1, num_envs, 1),
        )
        self.buffers["returns"] = torch.zeros(
            (num_steps + 1, num_envs, 1),
        )
        self.buffers["action_log_probs"] = torch.zeros(
            (num_steps + 1, num_envs, 1),
        )
        self.buffers["actions"] = torch.zeros(
            (num_steps + 1, num_envs, *action_shape),
        )
        self.buffers["prev_actions"] = torch.zeros(
            (num_steps + 1, num_envs, *action_shape),
        )
        self.buffers["masks"] = torch.zeros(
            (num_steps + 1, num_envs, 1),
            dtype=torch.bool,
        )

        # convert actions to long when discrete
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        # set initial values
        self._num_envs = num_envs
        self._num_steps = num_steps
        self.current_rollout_step_idx = 0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def to(self, device) -> None:
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
    ) -> None:

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if len(next_step) > 0:
            self.buffers.set(
                self.current_rollout_step_idx + 1,
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                self.current_rollout_step_idx,
                current_step,
                strict=False,
            )

    def advance_rollout(self) -> None:
        self.current_rollout_step_idx += 1

    def after_update(self) -> None:

        # put the last rollout in the beginning
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        # reset the rollout index
        self.current_rollout_step_idx = 0

    def compute_returns(self, next_value, use_gae, gamma, tau) -> None:
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))
