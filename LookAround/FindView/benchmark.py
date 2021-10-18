#!/usr/bin/env python3

from collections import defaultdict
from functools import partial
import json
import os
from os import PathLike
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.core.logging import logger
from LookAround.FindView.env import FindViewEnv
from LookAround.FindView.dataset import Episode
from LookAround.FindView.utils import generate_movement_video, obs2img
from LookAround.utils.visualizations import images_to_video


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

        # NOTE: before running the scripts, make sure to set the number of threads for numpy
        # correctly or else you will end up using all of the cores
        # os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
        # os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
        # os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
        # os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
        # os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

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
        torch.set_num_threads(self.bench_cfg.num_threads)
        cv2.setNumThreads(self.bench_cfg.num_threads)

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

        save_video = self.bench_cfg.save_video
        beautify = self.bench_cfg.beautify

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
        rot_histories = {}
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

            if save_video:
                if beautify:
                    renders = self.env.render(to_bgr=True)
                    pers = [renders['pers']]
                    target = renders['target']
                    actions = []
                else:
                    rgb_frames = []
                    rgb_frames.append(obs2img(**observations))

            while not self.env.episode_over:
                a_t = time.time()
                action = agent.act(observations)
                act_time += time.time() - a_t

                env_t = time.time()
                observations = self.env.step(action)
                env_time += time.time() - env_t

                if save_video:
                    # NOTE: the observation is still being added after stop is called
                    if beautify:
                        pers.append(self.env.render(to_bgr=True)['pers'])
                        actions.append(action)
                    else:
                        rgb_frames.append(obs2img(**observations))

            metrics = self.env.get_metrics()
            episodes_metrics.append(metrics)
            env_time = env_time / metrics['elapsed_steps']
            act_time = act_time / metrics['elapsed_steps']
            env_times.append(env_time)
            act_times.append(act_time)
            rot_history = self.env._rot_tracker.history
            rot_histories[current_episode.episode_id] = rot_history

            # save video to disk
            if save_video:
                if beautify:
                    # FIXME: OOM Killer (RAM) -> needs more than 32GB
                    # remove the last image to keep lengths consistent
                    pers.pop(-1)
                    if not self.env._called_stop:
                        rot_history.pop(-1)
                    assert len(pers) == len(actions)
                    assert len(pers) == len(rot_history)

                    pers_bboxs = []
                    for rot in rot_history:
                        pers_bboxs.append(self.env.sim.get_bounding_fov(rot))
                    target_bbox = self.env.sim.get_bounding_fov(current_episode.target_rotation)
                    video_name = (
                        f"beautify_episode-{current_episode.episode_id}_"
                        f"img-{current_episode.img_name}_"
                        f"difficulty-{current_episode.difficulty}_"
                        f"label-{current_episode.sub_label}"
                    )
                    generate_movement_video(
                        output_dir=self.video_dir,
                        video_name=video_name,
                        equi=self.env.sim.render_equi(to_bgr=True),
                        pers=pers,
                        target=target,
                        pers_bboxs=pers_bboxs,
                        target_bbox=target_bbox,
                        actions=actions,
                        add_text=False,
                        fps=30,
                    )
                else:
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
            ),
            rot_histories=rot_histories,
        )
        with open(self.metric_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        # logging
        for k, v in avg_metrics.items():
            logger.info("{}: {:.3f}".format(k, v))
        logger.info("env time: {:.3f}".format(env_time))
        logger.info("act time: {:.3f}".format(act_time))

        self.env.close()
