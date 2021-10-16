#!/usr/bin/env python3

"""
Agent that moves in the direction where the features matches the target image

FIXME:
- [ ] Choose feature descriptors (ORB, SIFT, etc)
- [ ] More robust movement generator
- [ ] Debug parameters for matching confidence
"""

import os

# Need to do this before the first numpy import
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

from collections import deque  # noqa
import random  # noqa
from statistics import mode  # noqa

import cv2  # noqa
import numpy as np  # noqa
import torch  # noqa

from LookAround.config import Config  # noqa
from LookAround.core.agent import Agent  # noqa
from LookAround.core.improc import (  # noqa
    post_process_for_render,
    post_process_for_render_torch,
)
from LookAround.FindView.actions import FindViewActions  # noqa


def movement_generator(size=4):
    """`size` is number of actions
    This movement generator moves around the initial point
    """
    idx = 0
    repeat = 1
    while True:
        for r in range(repeat):
            yield idx

        idx = (idx + 1) % size
        if idx % 2 == 0:
            repeat += 1


class FeatureMatchingAgent(Agent):

    detector = None
    matcher = None
    prev_action = None

    def __init__(
        self,
        feature_type: str = "ORB",
        matcher_type: str = "BF",
        knn_matching: bool = True,
        num_features: int = 500,
        num_matches: int = 10,
        distance_threshold: int = 30,
        stop_threshold: int = 5,
        num_track_actions: int = 50,
        num_threads: int = 1,
        seed: int = 0,
    ) -> None:

        self.name = 'fm'
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        for action in self.movement_actions:
            assert action in FindViewActions.all

        cv2.setNumThreads(num_threads)  # FIXME: doesn't really work...
        self.rst = random.Random(seed)

        # feature matching criteria
        self.num_matches = num_matches
        self.distance_threshold = distance_threshold
        self.stop_threshold = stop_threshold
        self.num_track_actions = num_track_actions

        # initialize detector
        if feature_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=num_features)
            norm_type = cv2.NORM_HAMMING
        elif feature_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=num_features)
            norm_type = cv2.NORM_L2
        else:
            raise ValueError()

        # initialize matcher
        if matcher_type == "BF":
            self.matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
        elif matcher_type == "FLANN":
            # FIXME: different params based on feature_type
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,  # 12
                key_size=12,  # 20
                multi_probe_level=1,  # 2
            )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError()
        self.matcher_type = matcher_type
        self.knn_matching = knn_matching

        # self.g = movement_generator(len(self.movement_actions)
        self.prev_action = self.rst.choice(self.movement_actions)
        self.tracked_actions = deque(maxlen=self.num_track_actions)

    @classmethod
    def from_config(cls, cfg: Config):
        agent_cfg = cfg.fm

        return cls(
            feature_type=agent_cfg.feature_type,
            matcher_type=agent_cfg.matcher_type,
            num_features=agent_cfg.num_features,
            num_matches=agent_cfg.num_matches,
            distance_threshold=agent_cfg.distance_threshold,
            stop_threshold=agent_cfg.stop_threshold,
            num_track_actions=agent_cfg.num_track_actions,
            num_threads=agent_cfg.num_threads,
        )

    def reset(self):
        # self.reset_movement_generator()
        if self.prev_action is None:
            self.prev_action = self.rst.choice(self.movement_actions)
        self.tracked_actions = deque(maxlen=self.num_track_actions)

    # def reset_movement_generator(self):
    #     self.g = movement_generator(len(self.movement_actions))

    def act(self, observations):

        pers = observations['pers']
        target = observations['target']

        # 1. Preprocess
        # preprocess the images to cv2 format
        if torch.is_tensor(pers):
            # NOTE: matching by allclose is slow
            # if torch.allclose(pers, target):
            #     return "stop"
            pers = post_process_for_render_torch(pers)
            target = post_process_for_render_torch(target)
        elif isinstance(pers, np.ndarray):
            # if np.allclose(pers, target):
            #     return "stop"
            pers = post_process_for_render(pers)
            target = post_process_for_render(target)
        else:
            raise ValueError("input image is not a valid type")

        # make it gray scale
        gray_pers = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # 2. detect features
        (kps_pers, des_pers) = self.detector.detectAndCompute(gray_pers, None)
        (kps_target, des_target) = self.detector.detectAndCompute(gray_target, None)

        if len(kps_pers) < self.num_matches or len(kps_target) < self.num_matches:
            # action = self.movement_actions[next(self.g)]
            action = self.prev_action
            return action

        # 3. Match features descriptors

        # find matches
        if self.knn_matching:
            # find knn matches
            if self.matcher_type == 'BF':
                raw_matches = self.matcher.knnMatch(des_pers, des_target, k=2)
                matches = []
                for m, n in raw_matches:
                    if m.distance < 0.75 * n.distance:
                        matches.append(m)
            elif self.matcher_type == 'FLANN':
                # FIXME: doesn't work...
                matches = self.matcher.knnMatch(kps_pers, kps_target, k=2)
                matchesMask = [[0, 0] for i in range(len(matches))]
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.7 * n.distance:
                        matchesMask[i] = [1, 0]
        else:
            matches = self.matcher.match(des_pers, des_target)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:self.num_matches]

        if len(matches) < self.num_matches:
            # print("not enough matches")
            # action = self.movement_actions[next(self.g)]
            action = self.prev_action
            return action

        # 4. Voting for actions
        actions = []
        for m in matches:

            # if m.distance > self.distance_threshold:
            #     continue

            pers_loc = np.float32(kps_pers[m.queryIdx].pt)
            target_loc = np.float32(kps_target[m.trainIdx].pt)

            x_displacement = np.abs(pers_loc[0] - target_loc[0])
            y_displacement = np.abs(pers_loc[1] - target_loc[1])

            if 0 <= x_displacement < self.stop_threshold and 0 <= y_displacement < self.stop_threshold:
                action = "stop"
            else:
                if x_displacement > y_displacement:
                    if pers_loc[0] > target_loc[0]:
                        action = "right"
                    elif pers_loc[0] < target_loc[0]:
                        action = "left"
                    else:
                        action = "stop"
                elif x_displacement < y_displacement:
                    if pers_loc[1] > target_loc[1]:
                        action = "down"
                    elif pers_loc[1] < target_loc[1]:
                        action = "up"
                    else:
                        action = "stop"
                else:
                    action = "stop"

            actions.append(action)

        if len(actions) == 0:
            # action = self.movement_actions[next(self.g)]
            # print("no actions")
            action = self.prev_action
            return action

        # 5. Post processes
        # self.reset_movement_generator()

        # tally up the votes and choose the best movement
        # NOTE: this only gets the first most common
        action = mode(actions)

        # append to deque
        self.tracked_actions.append(action)

        # if the tracked action consists of only opposites, it might mean that it's occilating
        # NOTE: make track actions large enough
        if len(self.tracked_actions) == self.num_track_actions:
            _what_actions = set(self.tracked_actions)
            if _what_actions == set(['right', 'left']):
                action = "stop"
            elif _what_actions == set(['up', 'down']):
                action = "stop"

        self.prev_action = action

        return action


def main():

    import argparse

    from LookAround.config import DictAction
    from LookAround.FindView.benchmark import FindViewBenchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--name',
        type=str,
        help='name of the agent (used for naming save directory)'
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='arguments in dict',
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    print(">>> Config:")
    print(cfg.pretty_text)

    # Initializing the agent
    agent = FeatureMatchingAgent.from_config(cfg)
    name = agent.name
    if args.name is not None:
        name += '_' + args.name

    # Benchmark
    print(f"Benchmarking {name}")
    benchmark = FindViewBenchmark(
        cfg=cfg,
        agent_name=name,
    )
    benchmark.evaluate(agent)


if __name__ == "__main__":
    main()
