#!/usr/bin/env python3

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

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


class FindViewBenchmark:

    def __init__(
        self,
        cfg: Config,
        device: torch.device = torch.device('cpu'),
    ) -> None:

        # FIXME: benchmark should have it's own configs

        # FIXME: change how we pass arguments
        # - for rl agents, we mostly want gpu and torch.float32 environment
        # - for baselines, we might want numpy environment
        # - we will also want to change how we set the filter
        self._env = FindViewEnv.from_config(
            cfg=cfg,
            split="test",
            filter_fn=partial(filter_by_difficulty, difficulties=['easy']),
            dtype=torch.float32,
            device=device,
        )

        # FIXME: since the evaluation is really long, maybe add a checkpoint?
        # FIXME: save the metrics as csv or json so I can compare by episode

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
            "l1_distance_to_target",
            "l2_distance_to_target",
            "num_same_view",
            "efficiency",
        )

        agg_metrics = defaultdict(float)
        for n in metric_names:
            for m in episode_data:
                agg_metrics[n] += m[n]

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
