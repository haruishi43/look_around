#!/usr/bin/env python3

"""Feature-matching agent parameter tuning

"""

import argparse
from collections import defaultdict
from copy import deepcopy
from functools import partial
import json
import os
import time
from typing import List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from LookAround.config import Config, DictAction
from LookAround.FindView.env import FindViewEnv
from LookAround.core.logging import logger
from LookAround.FindView.dataset import Episode

from findview_baselines.agents.feature_matching import FeatureMatchingAgent


def filter_by_difficulty(
    episode: Episode,
    difficulty: List[str],
) -> bool:
    return episode.difficulty in difficulty


def filter_out_sub_labels(
    episode: Episode,
    remove_labels: Union[List[str], Tuple[str]],
) -> bool:
    return episode.sub_label not in remove_labels


def joint_filter(
    episode: Episode,
    remove_labels: Union[List[str], Tuple[str]],
    difficulty: str,
) -> bool:
    return (
        filter_by_difficulty(episode, difficulty)
        and filter_out_sub_labels(episode, remove_labels)
    )


def single(
    cfg: Config,
    dataset_name: str,
    threshold: int,
    difficulty: str,
    remove_labels: List[str],
    num_episodes_per_img: int = 1,
    num_episodes: int = -1,
) -> float:
    filter_fn = partial(
        joint_filter,
        remove_labels=remove_labels,
        difficulty=difficulty,
    )
    env = FindViewEnv.from_config(
        cfg=cfg,
        split="val",  # NOTE: we are using validation split
        filter_fn=filter_fn,
        num_episodes_per_img=num_episodes_per_img,
        dtype=np.float32,
        device='cpu',
    )
    _cfg = deepcopy(cfg)
    _cfg.fm.distance_threshold = threshold
    print(cfg.fm.distance_threshold, _cfg.fm.distance_threshold)
    agent = FeatureMatchingAgent.from_config(_cfg)  # NOTE: make sure to use the edited Config

    # Name:
    name = f"fm_{cfg.fm.feature_type}"
    if threshold <= 0:
        name += '_inf'
    else:
        name += f'_{threshold}'

    save_path = os.path.join(
        'results',
        'fm_validations',
        dataset_name,
        f'{difficulty}_{name}.json',
    )
    parent_path = os.path.dirname(save_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    logger.info(f"Evaluating Agent {name} for {difficulty}")

    # Evaluation
    episodes_metrics = []
    env_times = []
    act_times = []

    if num_episodes <= 0:
        num_episodes = len(env.episodes)
    else:
        assert num_episodes > 0
    pbar = tqdm(total=num_episodes)
    count_episodes = 0
    while count_episodes < num_episodes:
        t = time.time()
        observations = env.reset()
        env_time = time.time() - t
        act_time = 0.
        agent.reset()

        while not env.episode_over:
            a_t = time.time()
            action = agent.act(observations)
            act_time += time.time() - a_t

            env_t = time.time()
            observations = env.step(action)
            env_time += time.time() - env_t

        metrics = env.get_metrics()
        episodes_metrics.append(metrics)
        env_time = env_time / metrics['elapsed_steps']
        act_time = act_time / metrics['elapsed_steps']
        env_times.append(env_time)
        act_times.append(act_time)

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
    )
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)

    # logging
    for k, v in avg_metrics.items():
        logger.info("{}: {:.3f}".format(k, v))
    logger.info("env time: {:.3f}".format(env_time))
    logger.info("act time: {:.3f}".format(act_time))

    env.close()

    return avg_metrics['l1_distance']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--diff",
        type=str,
        default='easy',
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help='number of episodes per evaluation',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='arguments in dict',
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    config: str,
    diff: str,
    num_episodes: int,
    options,
) -> None:
    cfg = Config.fromfile(config)
    if options is not None:
        cfg.merge_from_dict(options)
    print(">>> Config:")
    print(cfg.pretty_text)

    cv2.setNumThreads(cfg.fm.num_threads)

    # Params:
    remove_labels = ["others"]  # NOTE: hard-coded
    difficulty = diff
    num_episodes = num_episodes
    if cfg.dataset.name == 'sun360':
        dataset_name = "findview_{dataset}_{version}_{category}".format(
            dataset=cfg.dataset.name,
            version=cfg.dataset.version,
            category=cfg.dataset.category,
        )
    elif cfg.dataset.name == 'wacv360indoor':
        dataset_name = "findview_{dataset}_{version}".format(
            dataset=cfg.dataset.name,
            version=cfg.dataset.version,
        )
    else:
        raise ValueError()

    log_path = os.path.join(
        cfg.log_root,
        'fm_validations',
        dataset_name,
        f'{difficulty}_fm_{cfg.fm.feature_type}.log',
    )
    parent_path = os.path.dirname(log_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)
    logger.add_filehandler(log_path)

    metrics_by_thresholds = {}

    # Test infinite first
    distance = single(
        cfg=cfg,
        dataset_name=dataset_name,
        threshold=-1,
        difficulty=difficulty,
        remove_labels=remove_labels,
        num_episodes_per_img=1,
        num_episodes=num_episodes,
    )
    metrics_by_thresholds['inf'] = distance

    thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for threshold in thresholds:
        distance = single(
            cfg=cfg,
            dataset_name=dataset_name,
            threshold=threshold,
            difficulty=difficulty,
            remove_labels=remove_labels,
            num_episodes_per_img=1,
            num_episodes=num_episodes,
        )
        metrics_by_thresholds[threshold] = distance

    for threshold, distance in metrics_by_thresholds.items():
        logger.info(f"{threshold}: {distance}")

    best_threshold = min(metrics_by_thresholds, key=metrics_by_thresholds.get)
    logger.info(f"best threshold is... {best_threshold}")


if __name__ == "__main__":
    main()
