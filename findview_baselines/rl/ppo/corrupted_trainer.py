#!/usr/bin/env python3

import os
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, Optional

from mmengine import symlink

import torch
from torch.optim.lr_scheduler import LambdaLR

from LookAround.config import Config
from LookAround.core import logger
from LookAround.utils.random import seed
from LookAround.FindView.corrupted_vec_env import CorruptedVecEnv, construct_corrupted_envs

from findview_baselines.common import (
    BaseRLTrainer,
    TensorboardWriter,
    RolloutStorage,
)
from findview_baselines.common.scheduler import DifficultyScheduler, SeverityScheduler
from findview_baselines.rl.ppo import PPO, Policy
from findview_baselines.rl.ppo.corrupted_validator import CorruptedPPOValidator
from findview_baselines.rl.ppo.policy import FindViewBaselinePolicy
from findview_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    get_last_checkpoint_folder,
)


class CorruptedPPOTrainer(BaseRLTrainer):

    # Properties
    envs: Optional[CorruptedVecEnv]
    agent: Optional[PPO]
    actor_critic: Optional[Policy]

    # Hidden Properties
    _obs_batching_cache: ObservationBatchingCache

    def __init__(self, cfg: Config) -> None:
        assert cfg is not None, "ERR: needs config file to initialize trainer"
        super().__init__(cfg=cfg)

        # Initialize properties
        self.actor_critic = None
        self.agent = None
        self._obs_batching_cache = ObservationBatchingCache()

    def _init_rlenvs(
        self,
        cfg: Optional[Config] = None,
    ) -> None:
        if cfg is None:
            cfg = deepcopy(self.cfg)

        if cfg.trainer.dtype == "torch.float32":
            dtype = torch.float32
        elif cfg.trainer.dtype == "torch.float64":
            dtype = torch.float64
        else:
            raise ValueError()

        self.envs = construct_corrupted_envs(
            cfg=cfg,
            split="train",
            is_rlenv=True,
            dtype=dtype,
            device=self.device,
            vec_type=cfg.trainer.vec_type,
        )

    def _setup_actor_critic_agent(self) -> None:
        """Sets up actor critic and agent for PPO.
        """

        # FIXME: seems that more work is needed for setting seeds
        # torch.random.manual_seed(self.cfg.seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.cfg.seed)
        #     torch.backends.cudnn.deterministic = True  # type: ignore

        seed(self.cfg.seed)
        torch.random.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)  # this might not be necessary
            torch.backends.cudnn.deterministic = True  # type: ignore
            # torch.backends.cudnn.benchmark = False

        # FIXME: better to use registry for customization?
        self.actor_critic = FindViewBaselinePolicy(
            observation_space=self.obs_space,
            action_space=self.policy_action_space,
            **self.cfg.policy,
        )
        self.actor_critic.to(self.device)
        self.agent = PPO(
            actor_critic=self.actor_critic,
            **self.cfg.ppo,
        )

    def _collect_rollout_step(self):
        t_sample_action = time.time()

        # 1. sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[self.rollouts.current_rollout_step_idx]

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                observations=step_batch["observations"],
                rnn_hidden_states=step_batch["recurrent_hidden_states"],
                prev_actions=step_batch["prev_actions"],
                masks=step_batch["masks"],
                deterministic=False,
            )

        # move the actions to cpu
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        # 2. step environment
        t_step_env = time.time()
        outputs = self.envs.step([action.item() for action in actions.unbind(0)])
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]
        self.env_time += time.time() - t_step_env

        # 3. update stats
        t_update_stats = time.time()

        batch = batch_obs(
            observations=observations,
            device=self.device,
            cache=self._obs_batching_cache,
        )

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward += rewards
        current_ep_reward = self.current_episode_reward
        self.running_episode_stats["reward"] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward.masked_fill_(done_masks, 0.0)

        self.rollouts.insert(
            next_observations=batch,
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            rewards=rewards,
            next_masks=not_done_masks,
        )

        self.rollouts.advance_rollout()

        self.pth_time += time.time() - t_update_stats

        return self.rollouts.num_envs

    def _update_agent(self):
        ppo_cfg = self.cfg.ppo

        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                observations=step_batch["observations"],
                rnn_hidden_states=step_batch["recurrent_hidden_states"],
                prev_actions=step_batch["prev_actions"],
                masks=step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value=next_value,
            use_gae=ppo_cfg.use_gae,
            gamma=ppo_cfg.gamma,
            tau=ppo_cfg.tau,
        )

        # NOTE: set agent for training
        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self,
        losses: Dict[str, float],
        count_steps_delta: int,
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        self.num_steps_done += count_steps_delta

        return losses

    def _training_log(
        self,
        writer,
        losses: Dict[str, float],
        prev_time: int = 0,
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        # metrics = {
        #     k: v / deltas["count"]
        #     for k, v in deltas.items()
        #     if k not in {"reward", "count"}
        # }
        # if len(metrics) > 0:
        #     writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
        if self.num_updates_done % self.trainer_cfg.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            # log basic
            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            # log all deltas
            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def train(self) -> None:
        """Main method for training PPO.
        """

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.trainer_cfg.device)
        else:
            self.device = torch.device("cpu")

        ppo_cfg = self.cfg.ppo

        logger.add_filehandler(self.log_path)

        self._init_rlenvs()

        action_shape = None
        discrete_actions = True

        self._setup_actor_critic_agent()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        self.rollouts = RolloutStorage(
            num_steps=ppo_cfg.num_steps,
            num_envs=self.envs.num_envs,
            observation_space=self.obs_space,
            action_space=self.policy_action_space,
            recurrent_hidden_state_size=ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations=observations,
            device=self.device,
            cache=self._obs_batching_cache,
        )

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        count_checkpoints = 0
        prev_time = 0

        # this is only used when `use_linear_decay` is True
        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        # resumed from checkpoint
        if self.trainer_cfg.resume:
            if self.trainer_cfg.pretrained:
                logger.warn(f'{self.trainer_cfg.pretrained} would not be loaded since `resume` is `True`')

            ckpt_path = get_last_checkpoint_folder(self.ckpt_dir)
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

            # load model state
            state_dict = ckpt_dict.get('state_dict')
            assert state_dict is not None, f"ERR: {ckpt_path} doesn't have `state_dict`"
            self.agent.load_state_dict(state_dict)

            optim_state = ckpt_dict.get('optim_state')
            if optim_state is None:
                logger.warn(f'{ckpt_path} has no `optim_state`')
            else:
                self.agent.optimizer.load_state_dict(optim_state)

            lr_sched_state = ckpt_dict.get('lr_sched_state')
            if lr_sched_state is None:
                logger.warn(f'{ckpt_path} has no `lr_sched_state`')
            else:
                lr_scheduler.load_state_dict(lr_sched_state)

            extra_state: Dict[str, Any] = ckpt_dict.get("extra_state")
            if extra_state is None:
                logger.warn(f'{ckpt_path} has no `extra_state`; may impact stats')
            else:
                self.env_time = extra_state["env_time"]
                self.pth_time = extra_state["pth_time"]
                self.num_steps_done = extra_state["num_steps_done"]
                self.num_updates_done = extra_state["num_updates_done"]
                self._last_checkpoint_percent = extra_state[
                    "_last_checkpoint_percent"
                ]
                count_checkpoints = extra_state["count_checkpoints"]
                prev_time = extra_state["prev_time"]

                self.running_episode_stats = extra_state["running_episode_stats"]
                window_episode_stats = extra_state.get('window_episode_stats')
                if window_episode_stats is None:
                    logger.warn(f'{ckpt_path} has no `window_episode_stats`')
                else:
                    self.window_episode_stats.update(
                        extra_state["window_episode_stats"]
                    )

            logger.info(f"resuming from {ckpt_path} starting with {self.num_steps_done} steps")

        elif self.trainer_cfg.pretrained:
            assert os.path.exists(self.trainer_cfg.pretrained)
            ckpt_dict = self.load_checkpoint(self.trainer_cfg.pretrained, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict.get('state_dict'))

            logger.info(f"loading pretrained weights from {self.trainer_cfg.pretrained}")

        # account keeping stuff
        validator = CorruptedPPOValidator(cfg=self.cfg)
        difficulty_scheduler = DifficultyScheduler(
            initial_difficulty=self.cfg.scheduler.initial_difficulty,
            update_interval=self.cfg.scheduler.update_interval,
            num_updates_done=self.num_updates_done,
            bounded=self.cfg.dataset.bounded,
        )
        severity_scheduler = SeverityScheduler(
            initial_severity=self.cfg.corruption_scheduler.initial_severity,
            max_severity=self.cfg.corruption_scheduler.max_severity,
            update_interval=self.cfg.corruption_scheduler.update_interval,
        )
        checkpoint_distances = []

        with TensorboardWriter(self.tb_dir, flush_secs=self.flush_secs) as writer:
            while not self.is_done():

                # clip decay
                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                # gather rollouts
                self.agent.eval()
                count_steps_delta = 0
                for _ in range(self.rollouts.num_steps):
                    count_steps_delta += self._collect_rollout_step()

                # update agent
                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                # step learning rate
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        dist_entorpy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                # logging
                # FIXME: need to log the current difficulties of the episodes
                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if self.should_checkpoint():

                    # validate
                    # FIXME: difficulty changes so `is_best` is not reliable...
                    # FIXME: validation takes long when agent starts to not call stop
                    distance = validator.eval_from_trainer(
                        agent=self.agent,
                        writer=writer,
                        step_id=self.num_steps_done,
                        difficulty=difficulty_scheduler.current_difficulty,
                        bounded=difficulty_scheduler.bounded,
                        severity=severity_scheduler.current_severity,
                    )

                    # compare againt all checkpoints
                    is_best = False
                    if len(checkpoint_distances) > 0:
                        best_distance = min(checkpoint_distances)
                        is_best = distance < best_distance
                    else:
                        # FIXME: this doesn't work well when resumed
                        is_best = True
                    checkpoint_distances.append(distance)

                    extra_state = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )
                    state = dict(
                        state_dict=self.agent.state_dict(),
                        optim_state=self.agent.optimizer.state_dict(),
                        lr_sched_state=lr_scheduler.state_dict(),
                        cfg=self.cfg,
                    )
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        state,
                        extra_state=extra_state,
                    )

                    # link the best to ckpt.best.pth
                    # FIXME: need to add 'difficulty-aware' checkpoint mechanism
                    if is_best:
                        symlink(
                            os.path.join(os.path.abspath(self.ckpt_dir), f"ckpt.{count_checkpoints}.pth"),
                            os.path.join(os.path.abspath(self.ckpt_dir), "ckpt.best.pth"),
                        )

                    count_checkpoints += 1

                # update difficulty scheduler
                # FIXME: update difficulty scheduler based on metrics
                # FIXME: put this inside `if self.should_checkpoint()` loop?
                difficulty_scheduler.update(
                    envs=self.envs,
                    num_updates_done=self.num_updates_done,
                )
                severity_scheduler.update(
                    envs=self.envs,
                    num_updates_done=self.num_updates_done,
                )

            self.envs.close()
