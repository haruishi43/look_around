#!/usr/bin/env python3

import argparse
import os

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

    sun360_root = os.path.join(cfg.data_root, cfg.dataset_name)
    dataset_root = os.path.join(cfg.dataset_root, cfg.dataset_name, cfg.version, cfg.category)

    assert os.path.exists(sun360_root)
    assert os.path.exists(dataset_root)
