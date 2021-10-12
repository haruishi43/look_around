#!/usr/bin/env python3

from collections import defaultdict
from functools import partial
import json
import os
from os import PathLike
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.core.logging import logger
from LookAround.FindView.env import FindViewEnv
from LookAround.FindView.dataset import Episode
from LookAround.utils.visualizations import (
    images_to_video,
    obs2img,
)


def filter_by_difficulty(
    episode: Episode,
    difficulties: List[str],
) -> bool:
    return episode.difficulty in difficulties


def filter_out_sub_labels(
    episode: Episode,
    remove_labels: Union[List[str], Tuple[str]],
) -> bool:
    return episode.sub_label not in remove_labels


def joint_filter(
    episode: Episode,
    remove_labels: Union[List[str]],
    difficulties: List[str],
) -> bool:
    return (
        filter_by_difficulty(episode, difficulties)
        and filter_out_sub_labels(episode, remove_labels)
    )


class FindViewBenchmark(object):

    # Properties
    cfg: Config
    bench_cfg: Config
    env: FindViewEnv

    def __init__(
        self,
        cfg: Config,
        agent_name: str,
    ) -> None:

        self.cfg = cfg
        self.bench_cfg = cfg.benchmark

        # filter difficulties
        difficulty = self.bench_cfg.difficulty
        bounded = self.bench_cfg.bounded

        if bounded:
            difficulties = (difficulty,)
            bench_name = f"bounded_{difficulty}"
        else:
            if difficulty == "easy":
                difficulties = (difficulty,)
            elif difficulty == "medium":
                difficulties = ("easy", "medium")
            elif difficulty == "hard":
                difficulties = ("easy", "medium", "hard")
            else:
                raise ValueError()
            bench_name = f"unbounded_{difficulty}"

        for diff in difficulties:
            assert diff in ("easy", "medium", "hard")

        # filter sub labels
        remove_labels = self.bench_cfg.remove_labels
        if remove_labels is not None:
            if isinstance(remove_labels, str):
                remove_labels = [remove_labels]
            assert (
                len(remove_labels) > 0
                and (isinstance(remove_labels, list) or isinstance(remove_labels, tuple))
            )
            filter_fn = partial(
                joint_filter,
                remove_labels=remove_labels,
                difficulties=difficulties,
            )
        else:
            bench_name += "_all"  # identify when running all sub labels
            filter_fn = partial(filter_by_difficulty, difficulties=difficulties)

        # setting up environment
        if torch.cuda.is_available():
            device = torch.device(self.bench_cfg.device)
        else:
            device = torch.device("cpu")

        if cfg.benchmark.dtype == 'torch.float32':
            dtype = torch.float32
        elif cfg.benchmark.dtype == 'torch.float64':
            dtype = torch.float64
        elif cfg.benchmark.dtype == 'np.float32':
            dtype = np.float32
        elif cfg.benchmark.dtype == 'np.float64':
            dtype = np.float64
        else:
            raise ValueError()

        self.env = FindViewEnv.from_config(
            cfg=cfg,
            split="test",
            filter_fn=filter_fn,
            num_episodes_per_img=self.bench_cfg.num_episodes_per_img,
            dtype=dtype,
            device=device,
        )

        # setting bench info
        self.bench_name = bench_name
        assert len(agent_name) > 0
        self.agent_name = agent_name

    @property
    def video_dir(self) -> PathLike:
        video_dir = self.bench_cfg.video_dir.format(
            results_root=self.cfg.results_root,
            dataset=self.cfg.dataset.name,
            version=self.cfg.dataset.version,
            category=self.cfg.dataset.category,
            bench_name=self.bench_name,
            agent_name=self.agent_name,
        )
        assert '{' not in video_dir, video_dir
        assert '}' not in video_dir, video_dir
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        return video_dir

    @property
    def metric_path(self) -> PathLike:
        metric_path = self.bench_cfg.metric_path.format(
            results_root=self.cfg.results_root,
            dataset=self.cfg.dataset.name,
            version=self.cfg.dataset.version,
            category=self.cfg.dataset.category,
            bench_name=self.bench_name,
            agent_name=self.agent_name,
        )
        assert '{' not in metric_path, metric_path
        assert '}' not in metric_path, metric_path
        parent_path = os.path.dirname(metric_path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        return metric_path

    @property
    def log_path(self) -> PathLike:
        log_path = self.bench_cfg.log_file.format(
            log_root=self.cfg.log_root,
            dataset=self.cfg.dataset.name,
            version=self.cfg.dataset.version,
            category=self.cfg.dataset.category,
            bench_name=self.bench_name,
            agent_name=self.agent_name,
        )
        assert '{' not in log_path, log_path
        assert '}' not in log_path, log_path
        parent_path = os.path.dirname(log_path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        return log_path

    def evaluate_parallel(self):
        # TODO: since agents are made for single process, this might not be as efficient unless
        # we provide batch processing implementation for each agents.
        # Also, we can evaluate in batch using `validators`
        raise NotImplementedError

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
    ) -> None:

        if num_episodes is None:
            num_episodes = self.bench_cfg.num_episodes
            if num_episodes == -1:
                num_episodes = len(self.env.episodes)
        assert num_episodes > 0, "num_episodes should be greater than 0"
        assert num_episodes <= len(self.env.episodes), (
            "num_episodes({}) is larger than number of episodes "
            "in environment ({})".format(
                num_episodes, len(self.env.episodes)
            )
        )

        logger.add_filehandler(self.log_path)

        episodes_metrics = []
        env_times = []
        act_times = []

        pbar = tqdm(total=num_episodes)
        count_episodes = 0
        while count_episodes < num_episodes:
            t = time.time()
            observations = self.env.reset()
            env_time = time.time() - t
            act_time = 0.
            agent.reset()

            current_episode = self.env.current_episode
            rgb_frames = []
            rgb_frames.append(obs2img(**observations))

            while not self.env.episode_over:
                a_t = time.time()
                action = agent.act(observations)
                act_time += time.time() - a_t

                env_t = time.time()
                observations = self.env.step(action)
                env_time += time.time() - env_t

                rgb_frames.append(obs2img(**observations))

            metrics = self.env.get_metrics()
            episodes_metrics.append(metrics)
            env_time = env_time / metrics['elapsed_steps']
            act_time = act_time / metrics['elapsed_steps']
            env_times.append(env_time)
            act_times.append(act_time)

            # save video to disk
            video_name = (
                f"episode-{current_episode.episode_id}_"
                f"img-{current_episode.img_name}_"
                f"difficulty-{current_episode.difficulty}_"
                f"label-{current_episode.sub_label}"
            )
            images_to_video(
                images=rgb_frames,
                output_dir=self.video_dir,
                video_name=video_name,
                fps=10,
                quality=5,
                verbose=False,
            )

            count_episodes += 1
            pbar.update()

        assert count_episodes == num_episodes

        # select metrics to calculate
        metric_names = (
            "l1_distance",
            "l2_distance",
            "called_stop",
            "elapsed_steps",
            "num_same_view",
            "efficiency",
        )

        agg_metrics = defaultdict(float)
        for n in metric_names:
            for m in episodes_metrics:
                agg_metrics[n] += m[n]
        avg_metrics = {k: v / num_episodes for k, v in agg_metrics.items()}

        env_time = sum(env_times) / num_episodes
        act_time = sum(act_times) / num_episodes

        # save metrics
        save_dict = dict(
            summary=avg_metrics,
            episodes_metrics=episodes_metrics,
            times=dict(
                agent=act_time,
                env=env_time,
            )
        )
        with open(self.metric_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        # logging
        for k, v in avg_metrics.items():
            logger.info("{}: {:.3f}".format(k, v))
        logger.info("env time: {:.3f}".format(env_time))
        logger.info("act time: {:.3f}".format(act_time))

        self.env.close()
