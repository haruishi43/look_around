#!/usr/bin/env python3

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm

from LookAround.config import Config
from LookAround.core import logger
from LookAround.core.improc import post_process_for_render_torch
from LookAround.FindView.vec_env import VecEnv, construct_envs
from LookAround.utils.visualizations import renders_to_image

from findview_baselines.common.base_validator import BaseRLValidator
from findview_baselines.common.tensorboard_utils import TensorboardWriter
from findview_baselines.rl.ppo.ppo import PPO
from findview_baselines.rl.ppo.policy import FindViewBaselinePolicy
from findview_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
    generate_video,
)


class PPOValidator(BaseRLValidator):

    def __init__(self, cfg: Config) -> None:
        assert cfg is not None, "ERR: needs config file to initialize validator"
        super().__init__(cfg=cfg)

    def _init_rlenvs(
        self,
        split: str,
        cfg: Optional[Config] = None,
    ) -> VecEnv:
        if cfg is None:
            cfg = Config(deepcopy(self.cfg))
        split_cfg = getattr(cfg, split)
        assert split_cfg is not None
        envs = construct_envs(
            cfg=cfg,
            split=split,
            is_rlenv=True,
            dtype=exec(split_cfg.dtype),  # FIXME: potentially insecure
            device=torch.device(split_cfg.device),
            vec_type=split_cfg.vec_type,
        )
        # FIXME: change difficulties
        return envs

    def _eval_checkpoint(
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
        envs = self._init_rlenvs(
            split="test",
            cfg=self.cfg,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.cfg.test.device)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        torch.random.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        obs_space = envs.observation_spaces[0]
        policy_action_space = envs.action_spaces[0]

        actor_critic = FindViewBaselinePolicy(
            observation_space=obs_space,
            action_space=policy_action_space,
            **self.cfg.policy,
        )
        actor_critic.to(self.device)
        agent = PPO(
            actor_critic=actor_critic,
            **self.cfg.ppo,
        )
        agent.load_state_dict(ckpt_dict["state_dict"])

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        self._eval_single(agent, envs, writer=writer, step_id=step_id)

    def eval_from_trainer(
        self,
        agent: PPO,
        writer: Optional[TensorboardWriter],
        step_id: int,
    ):
        envs = self._init_rlenvs(
            split="val",
            cfg=self.cfg,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.cfg.val.device)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # FIXME: when the agent is on a different device during validation
        agent.actor_critic.to(self.device)

        self._eval_single(agent, envs, writer=writer, step_id=step_id)

    def _eval_single(
        self,
        agent: PPO,
        envs: VecEnv,
        writer: Optional[TensorboardWriter],
        step_id: int,
    ):

        actor_critic = agent.actor_critic

        action_shape = (1,)
        action_type = torch.long

        _obs_batching_cache = ObservationBatchingCache()
        observations = envs.reset()
        batch = batch_obs(
            observations,
            device=self.device,
            cache=_obs_batching_cache,
        )

        current_episode_reward = torch.zeros((envs.num_envs, 1), device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                envs.num_envs,
                actor_critic.net.num_recurrent_layers,
                self.cfg.ppo.hidden_size,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            (
                envs.num_envs,
                *action_shape,
            ),
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            (
                envs.num_envs,
                1,
            ),
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(envs.num_envs)
        ]  # type: List[List[np.ndarray]]

        # Add initial frames
        if len(self.video_option) > 0:
            for i in range(envs.num_envs):
                frame = renders_to_image(
                    {
                        k: post_process_for_render_torch(v[i], to_bgr=False)
                        for k, v in batch.items()
                    },
                )
                rgb_frames[i].append(frame)

        # get the number of episodes to test
        # FIXME: how many for validation???
        number_of_eval_episodes = self.cfg.test.episode_count

        # some checks for `number_of_eval_episodes`
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        # no backprop
        actor_critic.eval()

        while (
            len(stats_episodes) < number_of_eval_episodes
            and envs.num_envs > 0
        ):
            current_episodes = envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = actor_critic.act(
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

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=_obs_batching_cache,
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
            next_episodes = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
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

                    if len(self.video_option) > 0:
                        generate_video(
                            video_option=self.video_option,
                            video_dir=self.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=step_id,
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
                elif len(self.video_option) > 0:
                    frame = renders_to_image(
                        {
                            k: post_process_for_render_torch(v[i], to_bgr=False)
                            for k, v in batch.items()
                        },
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
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

        if (writer is not None) and isinstance(writer, TensorboardWriter):
            writer.add_scalars(
                "test_reward",
                {"average reward": aggregated_stats["reward"]},
                step_id,
            )

            metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
            if len(metrics) > 0:
                writer.add_scalars("test_metrics", metrics, step_id)

        envs.close()

        # FIXME: return metrics
