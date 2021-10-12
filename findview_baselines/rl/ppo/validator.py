#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm

from LookAround.config import Config
from LookAround.core import logger
from LookAround.FindView.vec_env import VecEnv
from LookAround.FindView.dataset.episode import Episode
from LookAround.utils.visualizations import obs2img

from findview_baselines.common.tensorboard_utils import TensorboardWriter
from findview_baselines.common.base_validator import BaseRLValidator
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

    def eval_from_trainer(
        self,
        agent: PPO,
        writer: Optional[TensorboardWriter],
        step_id: int,
        difficulty: str,
        bounded: bool,
    ) -> float:
        self.split = 'val'

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.cfg.trainer.device)
        else:
            self.device = torch.device("cpu")

        envs = self._init_rlenvs(
            difficulty=difficulty,
            bounded=bounded,
        )

        # FIXME: when the agent is on a different device during validation
        agent.actor_critic.to(self.device)
        agent.actor_critic.eval()

        return self._eval_single(
            agent=agent,
            envs=envs,
            writer=writer,
            step_id=step_id,
            num_episodes=self.val_cfg.num_episodes,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
    ) -> None:
        """Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.val_cfg.use_ckpt_cfg:
            loaded_cfg = ckpt_dict["cfg"]
            self.cfg.policy = loaded_cfg.policy
            self.cfg.ppo = loaded_cfg.ppo

        # if self.cfg.verbose:
        #     logger.info(self.cfg.pretty_text)

        # initialize the envs
        envs = self._init_rlenvs(
            difficulty=self.val_cfg.difficulty,
            bounded=self.val_cfg.bounded,
        )

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
        actor_critic.eval()
        agent = PPO(
            actor_critic=actor_critic,
            **self.cfg.ppo,
        )
        agent.load_state_dict(ckpt_dict["state_dict"])

        step_id = ckpt_dict["extra_state"]["num_steps_done"]

        self._eval_single(
            agent=agent,
            envs=envs,
            writer=writer,
            step_id=step_id,
            num_episodes=self.val_cfg.num_episodes,
        )

    def _eval_single(
        self,
        agent: PPO,  # FIXME: policy might be better
        envs: VecEnv,
        writer: Optional[TensorboardWriter],
        step_id: int,
        num_episodes: int,
    ) -> float:

        # isolate actor_critic
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

        # Add initial frames
        if len(self.video_option) > 0:
            rgb_frames = [
                [] for _ in range(envs.num_envs)
            ]  # type: List[List[np.ndarray]]

            for i in range(envs.num_envs):
                frame = obs2img(
                    pers=observations[i]['pers'],
                    target=observations[i]['target'],
                    to_bgr=False,
                )
                rgb_frames[i].append(frame)

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
        # dict of dicts that stores stats per episode
        stats_episodes: Dict[Any, Any, Any] = {}

        # save output to csv
        if self.val_cfg.save_metrics:
            episodes_stats = []

        # get the number of episodes to test
        number_of_eval_episodes = num_episodes
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
        logger.info(f"evaluating {number_of_eval_episodes}/{sum(envs.number_of_episodes)}")

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        while (
            len(stats_episodes) < number_of_eval_episodes
            and envs.num_envs > 0
        ):
            current_episodes: List[Episode] = envs.current_episodes()

            # 1. Sample actions
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

            # 2. Step environment
            step_data = [a.item() for a in actions.to(device="cpu")]
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards

            # 3. Reset when done
            for i, done in enumerate(dones):

                # add to `rgb_frames` for every step
                if len(self.video_option) > 0:
                    frame = obs2img(
                        pers=observations[i]['pers'],
                        target=observations[i]['target'],
                        to_bgr=False,
                    )
                    rgb_frames[i].append(frame)

                if done:
                    pbar.update()

                    # update episode_stats
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
                            current_episodes[i].difficulty,
                        )
                    ] = episode_stats

                    # NOTE: replace the observation when calling reset
                    observations[i] = envs.reset_at(i)

                    # save episode results
                    if self.val_cfg.save_metrics:
                        episodes_stats.append(infos[i])

                    # generate video
                    if len(self.video_option) > 0:
                        generate_video(
                            video_option=self.video_option,
                            video_dir=self.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=step_id,
                            metrics=infos[i],
                            tb_writer=writer,
                        )

                        # add first frame (since we just reset the env)
                        rgb_frames[i] = [
                            obs2img(
                                pers=observations[i]['pers'],
                                target=observations[i]['target'],
                                to_bgr=False,
                            )
                        ]

            # 4. Post process for policy's input
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
            not_done_masks = not_done_masks.to(device=self.device)

            # 5. Pause envs that repeats episodes
            next_episodes: List[Episode] = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                # detect if the episodes are repeating
                if (
                    next_episodes[i].img_name,
                    next_episodes[i].episode_id,
                    next_episodes[i].difficulty,
                ) in stats_episodes:
                    envs_to_pause.append(i)
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

        pbar.update()  # FIXME: might be redundant...

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        # log
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # save metrics
        if self.val_cfg.save_metrics:
            save_dict = dict(
                summary=aggregated_stats,
                episodes_stats=episodes_stats,
            )
            save_path = os.path.join(
                self.metric_dir,
                f"{step_id}_distance-{aggregated_stats['l1_distance']:.4f}.json",
            )
            with open(save_path, 'w') as f:
                json.dump(save_dict, f, indent=2)

        # tensorboard
        if (writer is not None) and isinstance(writer, TensorboardWriter):
            writer.add_scalars(
                "test_reward",
                {"average reward": aggregated_stats["reward"]},
                step_id,
            )

            metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
            if len(metrics) > 0:
                writer.add_scalars("test_metrics", metrics, step_id)

        # need to close the envs so memory is not pinned
        envs.close()

        # FIXME: debug
        # print(aggregated_stats)
        # >>> out: {
        #   'reward':,
        #   'elapsed_steps':,
        #   'called_stop':,
        #   'l1_distance':,
        #   'l2_distance':,
        #   'num_same_view':,
        #   'efficiency':,
        # }

        # NOTE: for now use l1 distance
        metric = aggregated_stats['l1_distance']

        return metric
