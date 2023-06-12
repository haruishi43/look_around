#!/usr/bin/env python3

"""Simple script for creating SUN360 Datset for FindView
"""

import argparse
import json
import os
import pickle
import random

from tqdm import tqdm

from LookAround.config import Config
from LookAround.utils.random import seed
from LookAround.FindView.dataset.sun360.helpers.beta import (
    gather_image_paths_in_category,
    create_splits_for_category,
    check_single_data_based_on_difficulty,
    make_single_data_based_on_difficulty,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="config file for creating dataset",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
    )
    parser.add_argument(
        "--check-dataset",
        action="store_true",
    )
    parser.add_argument(
        "--static-train",
        action="store_true",
    )
    parser.add_argument(
        "--no-shuffle",
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
    sun360_root = os.path.join(cfg.data_root, cfg.dataset.name)
    dataset_root = os.path.join(
        cfg.dataset_root,
        cfg.dataset.name,
        cfg.dataset.version,
        cfg.dataset.category,
    )
    split_ratios = cfg.dataset.split_ratios

    # check params
    seed(cfg.seed)  # NOTE: should be 0
    assert sum(split_ratios) == 1.0
    assert os.path.exists(sun360_root)
    assert os.path.exists(dataset_root)

    # splits paths
    # NOTE: using pickle just because it's easier for now
    # if we want to read what are in the `.data` files, that might
    # be a problem for now
    split_paths = {
        "train": os.path.join(dataset_root, "train.data"),
        "val": os.path.join(dataset_root, "val.data"),
        "test": os.path.join(dataset_root, "test.data"),
    }

    # 1. Create Splits (if needed)

    if args.rebuild_splits or not all(
        os.path.exists(p) for p in split_paths.values()
    ):
        print(f">>> BUILDING SPLITS for {cfg.dataset.category}")
        assert cfg.dataset.category in [
            "indoor",
            "outdoor",
        ]  # not using `others`
        category_paths = {
            "indoor": os.path.join(sun360_root, "indoor"),
            "outdoor": os.path.join(sun360_root, "outdoor"),
            "others": os.path.join(sun360_root, "others"),
        }

        category_path = os.path.join(sun360_root, cfg.dataset.category)
        assert os.path.exists(category_path)

        # get all image paths
        img_paths = gather_image_paths_in_category(
            category_path=category_path,
            data_root=sun360_root,
        )
        total = sum([len(p) for p in img_paths.values()])

        # make splits
        train, val, test = create_splits_for_category(img_paths, split_ratios)

        print("train:", len(train))
        print("val:", len(val))
        print("test:", len(test))
        print("all:", len(train) + len(val) + len(test))
        assert total == len(train) + len(val) + len(test)

        # save splits
        splits = {
            "train": train,
            "val": val,
            "test": test,
        }
        for name, values in splits.items():
            save_path = split_paths[name]
            with open(save_path, "wb") as f:
                pickle.dump(values, f)

        print("saved splits data file")

    # 2. Create Datasets

    # read splits from path
    print("reading splits data files...")
    splits = {}
    for name, path in tqdm(split_paths.items()):
        with open(path, "rb") as f:
            splits[name] = pickle.load(f)

    print("[train, val, test]", [len(s) for s in splits.values()])

    # check if the image exists
    for name, paths in splits.items():
        print(f"checking for images in {name} with {len(paths)} images")
        for path in tqdm(paths):
            assert os.path.exists(
                os.path.join(sun360_root, path)
            ), f"ERR: {path} doesn't exist! Did the dataset change?"

    # NOTE: make dataset for `val` and `test` only?
    # around >500mb for training json file
    # output format list of dict as (json)
    # [
    #   {
    #     "episode_id": <int>,
    #     "img_name": <string>,
    #     "path": <string>,
    #     "label": <string>,
    #     "sub_label": <string>,
    #     "initial_rot": {"roll": <int>, "pitch": <int>, "yaw": <int>},
    #     "target_rot": {"roll": <int>, "pitch": <int>, "yaw": <int>},
    #     "difficulty": <string>,
    #     "steps_for_shortest_path": <int>,
    #   }
    # ]

    use_splits = ["val", "test"]
    if args.static_train:
        print(">>> adding static train to this script")
        use_splits = ["train"] + use_splits

    # params from config
    fov = cfg.dataset.fov
    threshold = cfg.dataset.pitch_threshold
    num_easy = cfg.dataset.num_easy
    num_medium = cfg.dataset.num_medium
    num_hard = cfg.dataset.num_hard
    min_steps = cfg.dataset.min_steps
    max_steps = cfg.dataset.max_steps
    step_size = cfg.dataset.step_size
    mu = cfg.dataset.mu
    sigma = cfg.dataset.sigma
    sample_limit = cfg.dataset.sample_limit

    dataset_paths = {
        "train": os.path.join(dataset_root, "train.json"),
        "val": os.path.join(dataset_root, "val.json"),
        "test": os.path.join(dataset_root, "test.json"),
    }

    if args.check_dataset:
        # only check if the dataset created is valid

        for split_name in use_splits:
            print(f">>> checking data for {split_name}")
            dataset_path = dataset_paths[split_name]
            assert os.path.exists(dataset_path)

            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            assert len(dataset) > 0 and isinstance(dataset, list)

            id_count = 0  # iterating though list takes too much...
            for data in tqdm(dataset):
                eps_id = data["episode_id"]
                # assert eps_id == id_count
                id_count += 1

                name = data["img_name"]
                path = data["path"]
                assert os.path.exists(os.path.join(sun360_root, path))

                initial_rotation = data["initial_rotation"]
                target_rotation = data["target_rotation"]
                difficulty = data["difficulty"]
                short_step = data["steps_for_shortest_path"]

                check_single_data_based_on_difficulty(
                    initial_rotation=initial_rotation,
                    target_rotation=target_rotation,
                    difficulty=difficulty,
                    fov=fov,
                    short_step=short_step,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    step_size=step_size,
                    threshold=threshold,
                )

            # TODO: check for same/similar episodes
            # highly unlikely, but when the `step_size` becomes larger,
            # and test/val number per image increases,
            # there will be higher likelihood

    else:
        # create a new dataset
        for split_name in use_splits:
            # save path for dataset
            save_path = dataset_paths[split_name]
            if os.path.exists(save_path):
                print(
                    f"ALREADY HAVE DATASET for {split_name}; consider removing before continuing"
                )
                continue

            img_paths = splits[split_name]
            print(f">>> making dataset for {split_name} -> {len(img_paths)}")

            dataset = []
            # FIXME: make sure that it doesn't overflow
            eps_id = 0
            for img in tqdm(img_paths):
                assert os.path.exists(
                    os.path.join(sun360_root, img)
                ), f"{img} doesn't exist"

                # get cat and subcat
                # img_path is
                s = img.split("/")
                assert len(s) == 3, f"{img} is not valid"
                category = s[0]
                sub_category = s[1]

                base = {
                    "img_name": os.path.splitext(os.path.split(img)[-1])[0],
                    "path": img,
                    "label": category,
                    "sub_label": sub_category,
                }

                # FIXME: make sure that there won't be identical episodes
                for _ in range(num_easy):
                    # easy
                    difficulty = "easy"
                    data = make_single_data_based_on_difficulty(
                        difficulty=difficulty,
                        fov=fov,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        step_size=step_size,
                        threshold=threshold,
                        mu=mu,
                        sigma=sigma,
                        sample_limit=sample_limit,
                    )
                    dataset.append(
                        {**{"episode_id": eps_id}, **base, **data}
                    )  # update dict
                    eps_id += 1

                for _ in range(num_medium):
                    # medium
                    difficulty = "medium"
                    data = make_single_data_based_on_difficulty(
                        difficulty=difficulty,
                        fov=fov,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        step_size=step_size,
                        threshold=threshold,
                        mu=mu,
                        sigma=sigma,
                        sample_limit=sample_limit,
                    )
                    dataset.append(
                        {**{"episode_id": eps_id}, **base, **data}
                    )  # update dict
                    eps_id += 1

                for _ in range(num_hard):
                    # hard
                    difficulty = "hard"
                    data = make_single_data_based_on_difficulty(
                        difficulty=difficulty,
                        fov=fov,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        step_size=step_size,
                        threshold=threshold,
                        mu=mu,
                        sigma=sigma,
                        sample_limit=sample_limit,
                    )
                    dataset.append(
                        {**{"episode_id": eps_id}, **base, **data}
                    )  # update dict
                    eps_id += 1

            if not args.no_shuffle:
                print("shuffling")
                # FIXME: reorder episode_id
                random.shuffle(dataset)

            # dump to json
            # json_dataset = json.dumps(dataset, indent=4)  # turns to string
            with open(save_path, "w") as f:
                json.dump(dataset, f, indent=4)

            print(f"dumped {len(dataset)}")

    # if training mode is dynamic, we just save the essentials in `train.json`
    if not args.static_train and not args.check_dataset:
        print(
            ">>> making train.json with only essential items (img_name, etc...)"
        )

        # save path for train
        save_path = dataset_paths["train"]

        if os.path.exists(save_path):
            print(
                "ALREADY HAVE DATASET for train; consider removing before continuing"
            )
        else:
            train_imgs = splits["train"]

            dataset = []
            for img in tqdm(train_imgs):
                assert os.path.exists(
                    os.path.join(sun360_root, img)
                ), f"{img} doesn't exist"

                # get cat and subcat
                # img_path is
                s = img.split("/")
                assert len(s) == 3, f"{img} is not valid"
                category = s[0]
                sub_category = s[1]

                base = {
                    "img_name": os.path.splitext(os.path.split(img)[-1])[0],
                    "path": img,
                    "label": category,
                    "sub_label": sub_category,
                }
                dataset.append(base)

            if not args.no_shuffle:
                print("shuffling")
                # FIXME: reorder episode_id
                random.shuffle(dataset)

            with open(save_path, "w") as f:
                json.dump(dataset, f, indent=4)

            print(f"dumped {len(dataset)}")

    print("done")
