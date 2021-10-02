#!/usr/bin/env python3

import numpy as np

from LookAround.config import Config
from LookAround.FindView.rl_env import FindViewRLEnv, RLEnvRegistry


@RLEnvRegistry.register_module(name="Env1")
class FindViewRLEnv1(FindViewRLEnv):

    # FIXME: same as Basic for now

    def __init__(self, cfg: Config, **kwargs) -> None:

        self._rl_env_cfg = cfg.rl_env_cfgs
        self._slack_reward = self._rl_env_cfg.slack_reward
        self._success_reward = self._rl_env_cfg.success_reward
        self._end_reward_type = self._rl_env_cfg.end_type
        self._end_reward_param = self._rl_env_cfg.end_type_param

        super().__init__(cfg=cfg, **kwargs)

    def get_reward_range(self):
        # FIXME: better range calculation
        return (
            self._max_steps * self._slack_reward - (120 + 180),
            self._min_steps * self._slack_reward + self._success_reward,
        )

    def _reset_metrics(self) -> None:
        self._prev_dist = self._env.get_metrics()['l1_distance_to_target']

    def _end_rewards(self, measures):

        def bell_curve(x, threshold_steps):
            """Bell curve"""
            # NOTE: when x/threshold_steps == 1, the output is 0.36787944117144233
            return (np.e)**(-(x / threshold_steps)**2)

        reward_success = 0
        if self._env.episode_over and measures['called_stop']:
            l1 = measures['l1_distance_to_target']
            # l2 = measures['l2_distance_to_target']

            # FIXME: is success reward too high???
            # runs 1, 2:
            # reward_success = self._success_reward - l1
            # run 3:
            if self._end_reward_type == "inverse":
                reward_success = self._success_reward / (l1 + self._end_reward_param)
            # run 4: threshold = 10
            elif self._end_reward_type == "bell":
                reward_success = self._success_reward * bell_curve(l1, self._end_reward_param)  # FIXME parametrize
            else:
                raise ValueError("Reward function parameter is not set correctly")

        elif self._env.episode_over:
            # if agent couldn't finish by the limit, penalize them
            reward_success = -self._success_reward

        return reward_success

    def _same_view_penalty(self, measures):
        # Penality: Looked in the same spot
        # value is in the range of (0 ~ 1)
        num_same_rots = measures['num_same_view']
        reward_same_view = -num_same_rots
        return reward_same_view

    def get_reward(self, observations):
        # FIXME: make a good reward function here
        measures = self._env.get_metrics()
        curr_dist = measures['l1_distance_to_target']

        # Penalty: slack reward for every step it takes
        # value is small
        reward_slack = self._slack_reward

        # Reward/Penalty: if the agent is further from the target, penalize
        # value is either 1 or -1
        coef_dist = 0.1  # run_2=0.01, run1=1.0, etc...
        reward_dist = coef_dist * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        reward_success = self._end_rewards(measures)

        reward = reward_slack + reward_dist + reward_success

        return reward
