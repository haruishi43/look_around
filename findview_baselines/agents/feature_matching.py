#!/usr/bin/env python3

"""
Agent that moves in the direction where the features matches the target image
"""

from statistics import mode

import cv2
import numpy as np
import torch

from LookAround.config import Config
from LookAround.core.agent import Agent
from LookAround.core.improc import post_process_for_render, post_process_for_render_torch
from LookAround.FindView.actions import FindViewActions


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
    def __init__(
        self,
        cfg: Config,
    ) -> None:
        self.movement_actions = ["up", "right", "down", "left"]
        self.stop_action = "stop"
        for action in self.movement_actions:
            assert action in FindViewActions.all

        # feature matching criteria
        self.feature_type = cfg.fm.feature_type
        self.num_features = cfg.fm.num_features
        self.num_matches = cfg.fm.num_matches
        self.distance_threshold = cfg.fm.distance_threshold
        self.stop_threshold = cfg.fm.stop_threshold

        self.g = movement_generator(len(self.movement_actions))

    def reset(self):
        self.g = movement_generator(len(self.movement_actions))

    def act(self, observations):

        pers = observations['pers']
        target = observations['target']

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

        # extract features
        if self.feature_type == "ORB":
            detector = cv2.ORB_create(nfeatures=self.num_features)
            norm_type = cv2.NORM_HAMMING

        elif self.feature_type == "SIFT":
            detector = cv2.SIFT_create(nfeatures=self.num_features)
            norm_type = cv2.NORM_L2
        else:
            raise NotImplementedError

        (kps_pers, des_pers) = detector.detectAndCompute(gray_pers, None)
        (kps_target, des_target) = detector.detectAndCompute(gray_target, None)

        # FIXME: count kps to see if it has enough
        if len(kps_pers) < self.num_matches or len(kps_target) < self.num_matches:
            # print("not enough kp")
            return self.movement_actions[next(self.g)]

        # find matches
        # FIXME: flann or knn is better?
        matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
        matches = matcher.match(des_pers, des_target)

        # FIXME: filter only the 'good' matches
        if len(matches) < self.num_matches:
            # print("not enough matches")
            return self.movement_actions[next(self.g)]

        # FIXME: need to see if there are enough matches
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:self.num_matches]

        # vote direction
        actions = []
        for m in matches:

            # print(m.distance, m.queryIdx, m.trainIdx)

            # if m.distance > self.distance_threshold:
            #     continue

            # location in the perspective image
            # print(kps_pers[m.queryIdx].pt)
            # locatin in the target image
            # print(kps_target[m.trainIdx].pt)

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
            # print("no actions")
            return self.movement_actions[next(self.g)]

        # NOTE: reset generated movement
        self.reset()

        # tally up the votes and choose the best movement
        # NOTE: this only gets the first most common
        action = mode(actions)

        return action


def main():

    import argparse

    from LookAround.core.logging import logger
    from LookAround.FindView.benchmark import FindViewBenchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    agent = FeatureMatchingAgent(cfg)
    benchmark = FindViewBenchmark(
        cfg=cfg,
        device=torch.device('cpu'),
    )
    metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)

    for k, v in metrics.items():
        logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
