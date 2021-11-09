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
    threshold: int,
    difficulty: str,
    remove_labels: List[str],
    num_episodes_per_img: int = 1,
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
    agent = FeatureMatchingAgent.from_config(cfg)

    # Name:
    name = f"fm_{cfg.fm.feature_type}"
    if threshold <= 0:
        name += '_inf'
    else:
        name += f'_{threshold}'

    if _cfg.dataset.name == 'sun360':
        dataset_name = "findview_{dataset}_{version}_{category}".format(
            dataset=_cfg.dataset.name,
            version=_cfg.dataset.version,
            category=_cfg.dataset.category,
        )
    elif _cfg.dataset.name == 'wacv360indoor':
        dataset_name = "findview_{dataset}_{version}".format(
            dataset=_cfg.dataset.name,
            version=_cfg.dataset.version,
        )
    else:
        raise ValueError()

    save_path = os.path.join(
        'results',
        'fm_validations',
        dataset_name,
        f'{difficulty}_{name}.json',
    )
    parent_path = os.path.dirname(save_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)
    log_path = os.path.join(
        _cfg.log_root,
        'fm_validations',
        dataset_name,
        f'{difficulty}_{name}.log',
    )
    parent_path = os.path.dirname(log_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    # Evaluation
    logger.add_filehandler(log_path)
    episodes_metrics = []
    env_times = []
    act_times = []

    num_episodes = len(env.episodes)
    pbar = tqdm(total=num_episodes)
    count_episodes = 0

    logger.info(f"Evaluating Agent {name} for {difficulty}")
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

    single(
        cfg=cfg,
        threshold=-1,
        difficulty=difficulty,
        remove_labels=remove_labels,
        num_episodes_per_img=1,
    )

    print('end')


if __name__ == "__main__":
    main()
