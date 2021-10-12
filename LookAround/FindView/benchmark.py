#!/usr/bin/env python3

from collections import defaultdict
from functools import partial
from os import PathLike
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from LookAround.config import Config
from LookAround.core.agent import Agent

from LookAround.FindView.env import FindViewEnv
from LookAround.FindView.dataset import Episode


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
    bench_cfg: Config

    # Hidden Properties
    _env: FindViewEnv

    def __init__(
        self,
        cfg: Config,
        output_path: PathLike,
        video_path: PathLike,
        device: torch.device = torch.device('cpu'),
    ) -> None:

        # FIXME: benchmark should have it's own configs
        self.bench_cfg = cfg.benchmark

        # FIXME: change how we pass arguments
        # - for rl agents, we mostly want gpu and torch.float32 environment
        # - for baselines, we might want numpy environment
        # - we will also want to change how we set the filter

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

        if torch.cuda.is_available():
            device = torch.device("cuda", self.bench_cfg.device)
        else:
            device = torch.device("cpu")

        remove_labels = self.bench_cfg.remove_labels

        if remove_labels is not None:
            if isinstance(remove_labels, str):
                remove_labels = [remove_labels]
            assert (
                len(remove_labels) > 0
                and (isinstance(remove_labels, list) or isinstance(remove_labels, tuple))
            )
            self._env = FindViewEnv.from_config(
                cfg=cfg,
                split="test",
                filter_fn=partial(
                    joint_filter,
                    remove_labels=remove_labels,
                    difficulties=difficulties,
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            self._env = FindViewEnv.from_config(
                cfg=cfg,
                split="test",
                filter_fn=partial(filter_by_difficulty, difficulties=difficulties),
                dtype=torch.float32,
                device=device,
            )

    def evaluate_parallel(self):
        # FIXME: since agents are made for single process, this might not be as efficient unless
        # we provide batch processing implementation for each agents.
        # Also, we can evaluate in batch using `validators`
        raise NotImplementedError

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
    ) -> Dict[str, float]:

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        # NOTE: just save everything in a list first...
        # calculate benchmark results from this
        # also save the list results to find good qualitative test samples
        episode_data = []

        pbar = tqdm(total=num_episodes)
        count_episodes = 0
        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

            # FIXME: change the metrics to get
            # we can't use `string` metrics
            metrics = self._env.get_metrics()
            episode_data.append(metrics)
            count_episodes += 1
            pbar.update()

        # FIXME: save results to file? pkl, csv, etc...

        # FIXME: calculate average metrics
        # select metrics to calculate
        metric_names = (
            "l1_distance",
            "l2_distance",
            "num_same_view",
            "efficiency",
        )

        agg_metrics = defaultdict(float)
        for n in metric_names:
            for m in episode_data:
                agg_metrics[n] += m[n]

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
