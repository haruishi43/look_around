#!/usr/bin/env python3

"""Simple script to create dataset

- Test data should be static
- I think training data can be random, but we can create a training set just in case

First, I think we divide the dataset (SUN360) into 3 sets:
- Train
- Validation
- Test

I think we should use validation to check if the agent is not overfitting the training set

SUN360 classes:
- Indoor, outdoor, other
- Subclasses

What we need for our dataset:
- img_path
- initial rot
- target rot

Difficulties:
- easy (target is near initial rot; can see in FOV)
- medium (a little bit far; can see a little in FOV)
- hard (farther; can't see in FOV)

I think `difficulties` are used in scheduling to make it so that the agent can gradually
understand the task

Prior criteria:
- Need to make sure that the agent can reach the target
- Set yaw and pitch movements to increments of 1 degree (integers!)
- Using integers saves space too
- We also need to set the threshold of the upper and lower bound to pitch direction
- Since the FOV is usually 90 degrees, if you tilt 45 degrees, you can see the pole
- Maybe set the threshold to 60?
- Pitch should be normal distribution

# How...

## Create splits

## How to sample:

### Test and validation sets

1. Set hyper parameters (save to txt file or something so I don't forget)
2. Get all images

### Training Set

"""

import argparse
import json
import os
from os import PathLike
import pickle
import random
from typing import Dict, List

from mycv.utils.config import Config
import numpy as np

from LookAround.dataset.sampling import get_pitch_range, get_yaw_range


def seed(n: int):
    random.seed(n)
    np.random.seed(n)


def gather_image_paths_in_category(
    category_path: PathLike,
    data_root: PathLike,
) -> Dict[str, List[PathLike]]:
    assert os.path.exists(category_path)

    img_paths = {}

    # make sure that we only get directories
    subcats = [
        c for c in os.listdir(category_path)
        if os.path.isdir(os.path.join(category_path, c))
    ]
    # make a dictionary of paths for sub categories
    subcat_paths = {n: os.path.join(category_path, n) for n in subcats}

    for subcat_name, subcat_path in subcat_paths.items():
        subcat_img_names = [i for i in os.listdir(subcat_path) if os.path.splitext(i)[-1] == ".jpg"]
        subcat_img_paths = [os.path.relpath(os.path.join(subcat_path, i), data_root) for i in subcat_img_names]
        assert len(subcat_img_paths) > 0, f"ERR: {subcat_name} has no images"
        img_paths[subcat_name] = subcat_img_paths

    return img_paths


def create_splits_for_category(
    img_paths: Dict[str, List[str]],
    split_ratios: List[float],
) -> List[List[str]]:
    splits = [[] for _ in range(len(split_ratios))]
    for _, subcat_img_paths in img_paths.items():
        subcat_splits = make_splits(subcat_img_paths, split_ratios)

        for i, split in enumerate(subcat_splits):
            splits[i] += split

    return splits


def make_splits(
    img_paths: List[str],
    split_ratios: List[float],
) -> List[List[str]]:
    # NOTE: shuffle images!
    random.shuffle(img_paths)

    tot = len(img_paths)
    split_indices = []

    index = 0
    for r in split_ratios[:-1]:
        index += round(r * tot)
        split_indices.append(index)

    return [list(arr) for arr in np.split(img_paths, split_indices)]


def easy_condition(init_pitch, init_yaw, targ_pitch, targ_yaw):
    # check pitch
    # NOTE: should consider aspect ratio of the image, but
    # think that it's height is smaller than width
    diff_pitch = np.abs(init_pitch - targ_pitch)

    # check yaw
    diff_yaw = np.abs(init_yaw - targ_yaw)
    if diff_yaw > 180:
        diff_yaw = 360 - diff_yaw
    return np.sqrt(diff_yaw**2 + diff_pitch**2) <= np.sqrt(2 * ((fov / 2)**2))


def medium_condition(init_pitch, init_yaw, targ_pitch, targ_yaw):
    # check yaw
    diff_yaw = np.abs(init_yaw - targ_yaw)
    if diff_yaw > 180:
        diff_yaw = 360 - diff_yaw
    return (diff_yaw > fov / 2 and diff_yaw <= fov)


def hard_condition(init_pitch, init_yaw, targ_pitch, targ_yaw):
    # check yaw
    diff_yaw = np.abs(init_yaw - targ_yaw)
    if diff_yaw > 180:
        diff_yaw = 360 - diff_yaw
    return diff_yaw > fov


def make_single_data_based_on_difficulty(
    difficulty: str,
    fov: float,
    threshold: int = 60,
):
    # return initial and target rotations based on difficulty
    pitches, prob = get_pitch_range(threshold=threshold, mu=0.0, sigma=0.3)
    yaws = get_yaw_range()

    condition = None
    if difficulty == "easy":
        condition = easy_condition
    elif difficulty == "medium":
        condition = medium_condition
    elif difficulty == "hard":
        condition = hard_condition
    else:
        raise ValueError(f"ERR: unknown difficulty {difficulty}")

    assert condition is not None, "ERR something went horribly wrong here"

    _count = 0
    while True:
        # sample rotations
        init_pitch = int(np.random.choice(pitches, p=prob))
        init_yaw = int(np.random.choice(yaws))

        targ_pitch = int(np.random.choice(pitches, p=prob))
        targ_yaw = int(np.random.choice(yaws))

        if condition(init_pitch, init_yaw, targ_pitch, targ_yaw):
            return {
                "initial_rotation": {
                    "roll": 0,
                    "pitch": init_pitch,
                    "yaw": init_yaw,
                },
                "target_rotation": {
                    "roll": 0,
                    "pitch": targ_pitch,
                    "yaw": targ_yaw,
                },
                "difficulty": difficulty,
            }
        _count += 1

        if _count > 100000:
            raise RuntimeError("Raised because it's hard sampling with condition")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset"
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config_path = args.config
    assert os.path.exists(config_path)

    cfg = Config.fromfile(config_path)
    print(">>> Config:")
    print(cfg.pretty_text)

    # params
    sun360_root = os.path.join(cfg.data_root, "sun360")
    dataset_root = os.path.join(cfg.dataset_root, "sun360", cfg.version, cfg.category)
    split_ratios = cfg.splits

    # check params
    seed(cfg.seed)  # set seed
    assert sum(split_ratios) == 1.0
    assert os.path.exists(sun360_root)
    assert os.path.exists(dataset_root)

    # splits paths
    # NOTE: using pickle just because it's easier for now
    # if we want to read what's in the .data files, that might
    # be a problem for now
    split_paths = {
        "train": os.path.join(dataset_root, "train.data"),
        "val": os.path.join(dataset_root, "val.data"),
        "test": os.path.join(dataset_root, "test.data"),
    }

    # Create Splits (if needed)

    if args.rebuild_splits or not all(os.path.exists(p) for p in split_paths.values()):
        print(f">>> BUILDING SPLITS for {cfg.category}")
        assert cfg.category in ["indoor", "outdoor"]  # not using `others`
        category_paths = {
            "indoor": os.path.join(sun360_root, "indoor"),
            "outdoor": os.path.join(sun360_root, "outdoor"),
            "others": os.path.join(sun360_root, "others"),
        }

        category_path = os.path.join(sun360_root, cfg.category)
        assert os.path.exists(category_path)

        # get all image paths
        img_paths = gather_image_paths_in_category(
            category_path=category_path,
            data_root=sun360_root,
        )

        # make splits
        train, val, test = create_splits_for_category(img_paths, split_ratios)
        assert sum([len(p) for p in img_paths.values()]) == len(train) + len(val) + len(test)

        print("train:", len(train))
        print("val:", len(val))
        print("test:", len(test))
        print("all:", len(train) + len(val) + len(test))

        # save splits
        splits = {
            "train": train,
            "val": val,
            "test": test,
        }
        for name, values in splits.items():
            save_path = split_paths[name]
            with open(save_path, 'wb') as f:
                pickle.dump(values, f)

        print("saved")

    # Create Datasets

    # read splits from path
    splits = {}
    for name, path in split_paths.items():
        with open(path, 'rb') as f:
            splits[name] = pickle.load(f)

    print("[train, val, test]", [len(s) for s in splits.values()])

    # NOTE: make dataset for `val` and `test` only?
    # output format list of dict as (json)
    # [
    #   {
    #     "name": <img_name>,
    #     "path": <img_path>,
    #     "initial_rot": {"roll": 0., "pitch": 0., "yaw": 0.},
    #     "target_rot": {"roll": 0., "pitch": 0., "yaw": 0.},
    #     "difficulty": <string value>,
    #   }
    # ]

    # params from config
    fov = cfg.fov
    threshold = cfg.threshold_pitch
    num_easy = cfg.num_easy
    num_medium = cfg.num_medium
    num_hard = cfg.num_hard

    for split_name in ['val', 'test']:
        img_paths = splits[split_name]
        print(f"making dataset for {split_name} -> {len(img_paths)}")

        dataset = []
        for img in img_paths:
            assert os.path.exists(os.path.join(sun360_root, img)), f"{img} doesn't exist"

            base = {
                "name": os.path.split(img)[-1],
                "path": img,
            }

            for _ in range(num_easy):
                # easy
                difficulty = "easy"
                data = make_single_data_based_on_difficulty(
                    difficulty=difficulty,
                    fov=fov,
                    threshold=threshold,
                )
                dataset.append({**base, **data})  # update dict

            for _ in range(num_medium):
                # medium
                difficulty = "medium"
                data = make_single_data_based_on_difficulty(
                    difficulty=difficulty,
                    fov=fov,
                    threshold=threshold,
                )
                dataset.append({**base, **data})  # update dict

            for _ in range(num_hard):
                # hard
                difficulty = "hard"
                data = make_single_data_based_on_difficulty(
                    difficulty=difficulty,
                    fov=fov,
                    threshold=threshold,
                )
                dataset.append({**base, **data})  # update dict

        # dump to json
        save_path = os.path.join(dataset_root, f'{split_name}.json')
        with open(save_path, "w") as f:
            json.dump(dataset, f)

        print("dumped")

    print("done")
