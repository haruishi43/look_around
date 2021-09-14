#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, List, Optional

import torch

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
    ) -> None:

        # FIXME: change config?
        # FIXME: add filter?
        self._env = FindViewEnv(
            cfg=cfg,
            split='test',
            filter_fn=None,
            is_torch=True,
            dtype=torch.float32,
            device=torch.device('cpu'),
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

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

            # FIXME: change the metrics to get
            metrics = self._env.get_metrics()

            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
