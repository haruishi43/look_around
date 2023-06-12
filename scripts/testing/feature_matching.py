#!/usr/bin/env python3

"""
Trying out feature matching on OpenCV for feature matching agent

Input is the target image and a close image

- Compute feature matching
- Get coordinates of matched feature
- Compute population densities (where is it matching)
- Decide action (Move in yaw or pitch direction or stop)

"""

import argparse
import os

import cv2
import numpy as np
import torch

from LookAround.config import Config
from LookAround.FindView.sim import FindViewSim
from LookAround.FindView.rotation_tracker import RotationTracker

from findview_baselines.agents.feature_matching import FeatureMatchingAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="config file")
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # cfg = Config.fromfile(args.config)
    # print(f">>> Config:\n{cfg.pretty_text}")

    # params
    img_path = "./data/sun360/indoor/bedroom/pano_afvwdfmjeaglsd.jpg"
    initial_rots = {
        "roll": 0,
        "pitch": -20,
        "yaw": 30,
    }
    target_rots = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0,
    }

    num_steps = 2000
    dtype = torch.float32
    height = 256
    width = 256
    fov = 90.0
    sampling_mode = "bilinear"

    # initialize simulator
    sim = FindViewSim(
        height=height,
        width=width,
        fov=fov,
        sampling_mode=sampling_mode,
    )
    sim.inititialize_loader(
        dtype=dtype,
        device=torch.device("cpu"),
    )
    sim.load_episode(
        equi_path=img_path,
        initial_rotation=initial_rots,
        target_rotation=target_rots,
    )

    # add rotation tracker
    rot_tracker = RotationTracker(
        inc=1,
        pitch_threshold=60,
    )
    rot_tracker.initialize(initial_rots)

    # get two views
    # in this case just get the renders
    # render functions return images in numpy/cv2 format
    target = sim.render_target()
    pers = sim.render_pers()
    # render
    # cv2.imshow('target', target)
    # cv2.imshow('pers', pers)

    gray_pers = cv2.cvtColor(pers, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # extract features
    detector = cv2.ORB_create()
    (kps_pers, des_pers) = detector.detectAndCompute(gray_pers, None)
    tmp_pers = cv2.drawKeypoints(gray_pers, kps_pers, pers)

    (kps_target, des_target) = detector.detectAndCompute(gray_target, None)
    tmp_target = cv2.drawKeypoints(gray_target, kps_target, target)

    cv2.imshow("kps_pers", tmp_pers)
    cv2.imshow("kps_target", tmp_target)

    # match features
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    # bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    matches = bf.match(des_pers, des_target)
    # sort maches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    # limit matches
    matches = matches[:10]
    # draw matches
    tmp_match = cv2.drawMatches(
        gray_pers,
        kps_pers,
        gray_target,
        kps_target,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("maches", tmp_match)

    # analyze match locations

    # make a policy based on matched locations

    cv2.waitKey(0)
