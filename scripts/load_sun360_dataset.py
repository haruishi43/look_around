#!/usr/bin/env python3

import argparse
import json
import os

from tqdm import tqdm

from mycv.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config

    assert os.path.exists(config_path)

    cfg = Config.fromfile(config_path)
    print(">>> Config:")
    print(cfg.pretty_text)

    # basic params
    sun360_root = os.path.join(cfg.data_root, cfg.dataset_name)
    dataset_root = os.path.join(cfg.dataset_root, cfg.dataset_name, cfg.version, cfg.category)

    assert os.path.exists(sun360_root)
    assert os.path.exists(dataset_root)

    # params
    fov = cfg.fov
    threshold = cfg.threshold_pitch
    num_easy = cfg.num_easy
    num_medium = cfg.num_medium
    num_hard = cfg.num_hard
    min_steps = cfg.min_steps
    max_steps = cfg.max_steps
    step_size = cfg.step_size

    dataset_paths = {
        "train": os.path.join(dataset_root, "train.json"),
        "val": os.path.join(dataset_root, "val.json"),
        "test": os.path.join(dataset_root, "test.json"),
    }

    some_split = "val"

    print(f"checking data for {some_split}")
    dataset_path = dataset_paths[some_split]
    assert os.path.exists(dataset_path)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    assert len(dataset) > 0 and isinstance(dataset, list)

    for data in tqdm(dataset):
        name = data["img_name"]
        path = data["path"]
        initial_rotation = data["initial_rotation"]
        target_rotation = data["target_rotation"]
        difficulty = data["difficulty"]
        eps_id = data["episode_id"]
