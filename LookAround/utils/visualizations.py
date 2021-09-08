#!/usr/bin/env python3

from typing import List

import cv2

import numpy as np


def save_images_as_video(images, video_path):
    assert len(images) > 0, \
        "No images in list"
    img = images[0]
    height, width = img.shape[:-1]

    # NOTE: figure out other formats later
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    for img in images:
        video.write(img)

    video.release()
    cv2.destroyAllWindows()


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    """Tile multiple images into single image
    Args:
        images: list of images where each image has dimension
            (height x width x channels)
    Returns:
        tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape(
        new_height, new_width, height, width, n_channels
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image
