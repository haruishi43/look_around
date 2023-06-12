#!/usr/bin/env python3

from functools import partial
import os
from os import PathLike
import random
from typing import Any, Dict, List

import numpy as np

from LookAround.FindView.dataset.sampling import (
    base_condition,
    easy_condition,
    find_minimum,
    medium_condition,
    hard_condition,
    get_pitch_range,
    get_yaw_range,
)


def gather_image_paths(
    img_root: PathLike,
    data_root: PathLike,
) -> List[PathLike]:
    assert os.path.exists(img_root)
    img_names = [
        i for i in os.listdir(img_root) if os.path.splitext(i)[-1] == ".jpg"
    ]
    img_paths = [
        os.path.relpath(os.path.join(img_root, i), data_root) for i in img_names
    ]
    assert len(img_paths) > 0, "ERR: no images"
    return img_paths


def create_splits(
    img_paths: List[PathLike],
    split_ratios: List[float],
) -> List[List[PathLike]]:
    total = len(img_paths)

    # NOTE: shuffle images!
    random.shuffle(img_paths)
    split_indices = []
    index = 0
    for r in split_ratios[:-1]:
        index += round(r * total)
        split_indices.append(index)

    train, val, test = [list(arr) for arr in np.split(img_paths, split_indices)]
    return [train, val, test]


def make_single_data_based_on_difficulty(
    difficulty: str,
    fov: float,
    min_steps: int,
    max_steps: int,
    step_size: int,
    threshold: int = 60,
    mu: float = 0.0,
    sigma: float = 0.3,
    sample_limit: int = 100000,
) -> Dict[str, Any]:
    # FIXME: Optimize since it does take a bit of time...

    # return initial and target rotations based on difficulty
    pitches, prob = get_pitch_range(threshold=threshold, mu=mu, sigma=sigma)
    yaws = get_yaw_range()

    base_cond = partial(
        base_condition,
        min_steps=min_steps,
        max_steps=max_steps,
        step_size=step_size,
    )

    cond = None
    if difficulty == "easy":
        cond = partial(easy_condition, fov=fov)
    elif difficulty == "medium":
        cond = partial(medium_condition, fov=fov)
    elif difficulty == "hard":
        cond = partial(hard_condition, fov=fov)
    else:
        raise ValueError(f"ERR: unknown difficulty {difficulty}")

    assert cond is not None, "ERR something went horribly wrong here"

    _count = 0  # FIXME: how to deal with criteria that's REALLY hard?
    while True:
        # sample rotations
        init_pitch = int(np.random.choice(pitches, p=prob))
        init_yaw = int(np.random.choice(yaws))

        targ_pitch = int(np.random.choice(pitches, p=prob))
        targ_yaw = int(np.random.choice(yaws))

        if base_cond(init_pitch, init_yaw, targ_pitch, targ_yaw) and cond(
            init_pitch, init_yaw, targ_pitch, targ_yaw
        ):
            # shortest path
            diff_pitch = abs(init_pitch - targ_pitch)
            diff_yaw = find_minimum(abs(init_yaw - targ_yaw))
            shortest_path = int(
                diff_pitch + diff_yaw
            )  # NOTE: includes `stop` action

            return dict(
                initial_rotation=dict(
                    roll=0,
                    pitch=init_pitch,
                    yaw=init_yaw,
                ),
                target_rotation=dict(
                    roll=0,
                    pitch=targ_pitch,
                    yaw=targ_yaw,
                ),
                difficulty=difficulty,
                steps_for_shortest_path=shortest_path,
            )
        _count += 1

        if _count > sample_limit:
            raise RuntimeError(
                "Raised because it's hard sampling with condition, try making the condition easier"
            )


def check_single_data_based_on_difficulty(
    initial_rotation: Dict[str, int],
    target_rotation: Dict[str, int],
    difficulty: str,
    fov: float,
    min_steps: int,
    max_steps: int,
    step_size: int,
    short_step: int,
    threshold: int = 60,
) -> None:
    # check input
    for d in initial_rotation.values():
        assert isinstance(d, int)
    for d in target_rotation.values():
        assert isinstance(d, int)

    init_pitch = initial_rotation["pitch"]
    init_yaw = initial_rotation["yaw"]
    targ_pitch = target_rotation["pitch"]
    targ_yaw = target_rotation["yaw"]

    assert (np.abs(init_pitch) <= threshold) and (
        np.abs(targ_pitch) <= threshold
    )

    # check basic condition
    assert base_condition(
        init_pitch,
        init_yaw,
        targ_pitch,
        targ_yaw,
        min_steps,
        max_steps,
        step_size,
    )

    # check difficulty condition
    if difficulty == "easy":
        ret = easy_condition(
            init_pitch,
            init_yaw,
            targ_pitch,
            targ_yaw,
            fov,
        )
    elif difficulty == "medium":
        ret = medium_condition(
            init_pitch,
            init_yaw,
            targ_pitch,
            targ_yaw,
            fov,
        )
    elif difficulty == "hard":
        ret = hard_condition(
            init_pitch,
            init_yaw,
            targ_pitch,
            targ_yaw,
            fov,
        )
    else:
        raise ValueError(f"{difficulty} is not supported")
    assert ret, (
        "ERR: failed check with ({init_pitch}, {init_yaw}) "
        "({targ_pitch}, {targ_yaw}) for {difficulty}"
    ).format(
        init_pitch=init_pitch,
        init_yaw=init_yaw,
        targ_pitch=targ_pitch,
        targ_yaw=targ_yaw,
        difficulty=difficulty,
    )

    # NOTE: includes `stop` action
    # shortest path
    diff_pitch = abs(init_pitch - targ_pitch)
    diff_yaw = find_minimum(abs(init_yaw - targ_yaw))
    shortest_path = int(diff_pitch + diff_yaw)  # NOTE: includes `stop` action

    assert shortest_path == short_step
