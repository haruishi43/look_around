#!/usr/bin/env python3

import cv2


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
