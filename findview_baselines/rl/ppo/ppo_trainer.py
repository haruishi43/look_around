#!/usr/bin/env python3

from copy import deepcopy
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from LookAround.config import Config
from LookAround.core import logger
from LookAround.core.improc import post_process_for_render_torch
from LookAround.FindView import ThreadedVecEnv, FindViewEnv, FindViewRLEnv
from LookAround.utils.visualizations import renders_to_image

# FIXME: move this to findview_baselines?
from LookAround.FindView.vec_env import construct_envs, construct_test_envs

from findview_baselines.common.rollout_storage import RolloutStorage
from findview_baselines.common.tensorboard_utils import TensorboardWriter

from findview_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
    get_checkpoint_id,
    get_last_checkpoint_folder,
    poll_checkpoint_folder,
)

from findview_baselines.rl.ppo import PPO, Policy
from findview_baselines.rl.ppo.policy import FindViewBaselinePolicy


class PPOTrainer:

    cfg: Config
    device: torch.device  # type: ignore
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _obs_batching_cache: ObservationBatchingCache
    envs: ThreadedVecEnv
    agent: PPO
    actor_critic: Policy

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg is not None, "ERR: needs config file to initialize trainer"
        self.cfg = cfg
        self._flush_secs = 30
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._last_checkpoint_percent = -1.0

        if cfg.num_updates != -1 and cfg.total_num_steps != -1:
            raise RuntimeError(
                "`num_updates` and `total_num_steps` are both specified. One must be -1.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    cfg.num_updates, cfg.total_num_steps,
                )
            )
        if cfg.num_updates == -1 and cfg.total_num_steps == -1:
            raise RuntimeError(
                "One of `num_updates` and `total_num_steps` must be specified.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    cfg.num_updates, cfg.total_num_steps,
                )
            )
        if cfg.num_ckpts != -1 and cfg.ckpt_interval != -1:
            raise RuntimeError(
                "`num_ckpts` and `ckpt_interval` are both specified."
                "  One must be -1.\n"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    cfg.num_ckpts, cfg.ckpt_interval
                )
            )

        if cfg.num_ckpts == -1 and cfg.ckpt_interval == -1:
            raise RuntimeError(
                "One of `num_ckpts` and `ckpt_interval` must be specified"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    cfg.num_ckpts, cfg.ckpt_interval
                )
            )

        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        self._obs_batching_cache = ObservationBatchingCache()

    def percent_done(self) -> float:
        if self.cfg.num_updates != -1:
            return self.num_updates_done / self.cfg.num_updates
        else:
            return self.num_steps_done / self.cfg.total_num_steps

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.cfg.num_ckpts != -1:
            checkpoint_every = 1 / self.cfg.num_ckpts
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done % self.cfg.ckpt_interval
            ) == 0

        return needs_checkpoint

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    @property
    def ckpt_dir(self):
        ckpt_dir = self.cfg.ckpt_dir.format(
            results_root=self.cfg.results_root,
            run_id=str(self.cfg.run_id),
        )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @property
    def tb_dir(self):
        tb_dir = self.cfg.tb_dir.format(
            tb_root=self.cfg.tb_root,
            run_id=str(self.cfg.run_id),
        )
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        return tb_dir

    @property
    def video_dir(self):
        video_dir = self.cfg.video_dir.format(
            results_root=self.cfg.results_root,
            run_id=self.cfg.run_id,
        )
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        return video_dir

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[ThreadedVecEnv, FindViewRLEnv, FindViewEnv],
        test_recurrent_hidden_states: torch.Tensor,
        not_done_masks: torch.Tensor,
        current_episode_reward: torch.Tensor,
        prev_actions: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        rgb_frames: Union[List[List[Any]], List[List[np.ndarray]]],
    ) -> Tuple[
        Union[ThreadedVecEnv, FindViewRLEnv, FindViewEnv],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _setup_actor_critic_agent(self) -> None:
        """Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """

        # random.seed(0)
        torch.random.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        # FIXME: add tests in Policy and PPO agent
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

    def _init_envs(
        self,
        split: str,
        cfg: Optional[Config] = None,
    ) -> None:
        if cfg is None:
            cfg = Config(deepcopy(self.cfg))
        split_cfg = getattr(cfg, split)
        assert split_cfg is not None
        if split_cfg.is_torch:
            dtype = torch.float32
        else:
            dtype = np.float32
        self.envs = construct_envs(
            env_cls=FindViewRLEnv,
            cfg=cfg,
            split=split,
            is_torch=split_cfg.is_torch,
            dtype=dtype,
            device=torch.device(split_cfg.device),
            vec_type="threaded",
        )

    def _init_test_envs(
        self,
        split: str,
        difficulties: List[str] = ["easy"],
        cfg: Optional[Config] = None,
    ) -> None:
        if cfg is None:
            cfg = Config(deepcopy(self.cfg))
        assert split == "test"  # FIXME: for now...
        split_cfg = getattr(cfg, split)
        assert split_cfg is not None
        if split_cfg.is_torch:
            dtype = torch.float32
        else:
            dtype = np.float32
        self.envs = construct_test_envs(
            env_cls=FindViewRLEnv,
            cfg=cfg,
            split=split,
            difficulties=difficulties,
            is_torch=split_cfg.is_torch,
            dtype=dtype,
            device=torch.device(split_cfg.device),
            vec_type="threaded",
        )

    def save_checkpoint(
        self, file_name: str,
        save_state: Dict,
        extra_state: Optional[Dict] = None
    ) -> None:
        """Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """

        if extra_state is not None:
            save_state["extra_state"] = extra_state

        torch.save(
            save_state, os.path.join(self.ckpt_dir, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = [
        "episode_id",
        "initial_rotation",
        "target_rotation",
        "current_rotation",
        "steps_for_shortest_path",
    ]

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            step_action = act.item()

            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
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

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
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

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    def _update_agent(self):
        ppo_cfg = self.cfg.ppo
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

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
        self, losses: Dict[str, float], count_steps_delta: int
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
        self, writer, losses: Dict[str, float], prev_time: int = 0
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
        if self.num_updates_done % self.cfg.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

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
        """Main method for training DD/PPO.
        Returns:
            None
        """

        logger.add_filehandler(
            self.cfg.log_file.format(
                split="train",
                log_root=self.cfg.log_root,
                run_id=self.cfg.run_id,
            )
        )

        self._init_envs(split="train")

        self.policy_action_space = self.envs.action_spaces[0]
        action_shape = None
        discrete_actions = True

        ppo_cfg = self.cfg.ppo
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.cfg.train.device)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self._setup_actor_critic_agent()

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
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

        # this is only used when `user_linear_decay` is True
        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        if self.cfg.train.resume:
            ckpt_path = get_last_checkpoint_folder(self.ckpt_dir)
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

            # load model state
            state_dict = ckpt_dict.get('state_dict')
            assert state_dict is not None, f"ERR: {ckpt_path} doesn't have `state_dict`"
            self.agent.load_state_dict(state_dict)

            # FIXME: old code didn't save optim_state and lr_sched_state
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

            extra_state = ckpt_dict.get("extra_state")
            if extra_state is None:
                logger.warn(f'{ckpt_path} has no `extra_state`; may impact stats')
            else:
                # FIXME: change to `extra_states`
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

        with TensorboardWriter(self.tb_dir, flush_secs=self.flush_secs) as writer:
            while not self.is_done():
                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                self.agent.eval()
                count_steps_delta = 0

                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (step + 1) == ppo_cfg.num_steps

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if not is_last_step:
                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

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

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if self.should_checkpoint():
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
                    count_checkpoints += 1

            self.envs.close()

    def test(self) -> None:
        """Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer
        Returns:
            None

        NOTE: make sure that `num_envs=1` inorder to get the most consistent results
        """

        logger.add_filehandler(
            self.cfg.log_file.format(
                split="test",
                log_root=self.cfg.log_root,
                run_id=self.cfg.run_id,
            )
        )

        self.device = torch.device(self.cfg.test.device)

        with TensorboardWriter(
            self.tb_dir, flush_secs=self.flush_secs
        ) as writer:
            ckpt_path = os.path.join(self.ckpt_dir, self.cfg.test.ckpt_path)
            if os.path.isfile(ckpt_path):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(ckpt_path)
                assert proposed_index is not None, \
                    f"ERR: could not find valid ckpt for {ckpt_path}"
                ckpt_idx = proposed_index
                self._test_checkpoint(
                    ckpt_path,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = poll_checkpoint_folder(
                        self.ckpt_dir, prev_ckpt_ind
                    )
                    if current_ckpt is None:
                        break
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._test_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _test_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.cfg.test.use_ckpt_cfg:
            loaded_cfg = ckpt_dict["cfg"]
        else:
            loaded_cfg = deepcopy(self.cfg)

        self.cfg.policy = loaded_cfg.policy
        self.cfg.ppo = loaded_cfg.ppo

        # if self.cfg.verbose:
        #     logger.info(self.cfg.pretty_text)

        # initialize the envs
        self._init_test_envs(
            split="test",
            difficulties=["easy"],
            cfg=self.cfg,
        )

        self.policy_action_space = self.envs.action_spaces[0]
        action_shape = (1,)
        action_type = torch.long

        self._setup_actor_critic_agent()

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.envs.num_envs,
            self.actor_critic.net.num_recurrent_layers,
            self.cfg.ppo.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.envs.num_envs,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.envs.num_envs)
        ]  # type: List[List[np.ndarray]]

        # Add initial frames
        if len(self.cfg.video_option) > 0:
            for i in range(self.envs.num_envs):
                frame = renders_to_image(
                    {
                        k: post_process_for_render_torch(v[i], to_bgr=False)
                        for k, v in batch.items()
                    },
                )
                rgb_frames[i].append(frame)

        # get the number of episodes to test
        number_of_eval_episodes = self.cfg.test.episode_count

        # some checks for `number_of_eval_episodes`
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].img_name,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()

                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use img_name + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].img_name,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.cfg.video_option) > 0:
                        generate_video(
                            video_option=self.cfg.video_option,
                            video_dir=self.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        # add first frame
                        rgb_frames[i] = [
                            renders_to_image(
                                {
                                    k: post_process_for_render_torch(v[i], to_bgr=False)
                                    for k, v in batch.items()
                                },
                            )
                        ]

                # episode continues
                elif len(self.cfg.video_option) > 0:
                    frame = renders_to_image(
                        {
                            k: post_process_for_render_torch(v[i], to_bgr=False)
                            for k, v in batch.items()
                        },
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.update()  # debug

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "test_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("test_metrics", metrics, step_id)

        self.envs.close()
