#!/usr/bin/env python3

"""Reference:
https://github.com/facebookresearch/habitat-lab/blob/master/habitat/utils/visualizations/utils.py
"""

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import tqdm

from LookAround.core import logger
from LookAround.core.improc import post_process_for_render_torch


def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    """Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    """Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    if verbose:
        logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()


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


def cleaner_tile_images(render_obs_images: List[np.ndarray]) -> np.ndarray:
    """Tiles multiple images of non-equal size to a single image. Images are
    tiled into columns making the returned image wider than tall.
    """
    # Get the images in descending order of vertical height.
    render_obs_images = sorted(
        render_obs_images, key=lambda x: x.shape[0], reverse=True
    )
    img_cols = [[render_obs_images[0]]]
    max_height = render_obs_images[0].shape[0]
    cur_y = 0.0
    # Arrange the images in columns with the largest image to the left.
    col = []
    for im in render_obs_images[1:]:
        if cur_y + im.shape[0] <= max_height:
            col.append(im)
            cur_y += im.shape[0]
        else:
            img_cols.append(col)
            col = [im]
            cur_y = im.shape[0]
    img_cols.append(col)
    col_widths = [max(col_ele.shape[1] for col_ele in col) for col in img_cols]
    # Get the total width of all the columns put together.
    total_width = sum(col_widths)

    # Tile the images, pasting the columns side by side.
    final_im = np.zeros(
        (max_height, total_width, 3), dtype=render_obs_images[0].dtype
    )
    cur_x = 0
    for i in range(len(img_cols)):
        next_x = cur_x + col_widths[i]
        total_col_im = np.concatenate(img_cols[i], axis=0)
        final_im[: total_col_im.shape[0], cur_x:next_x] = total_col_im
        cur_x = next_x
    return final_im


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


def obs2img(
    pers: torch.Tensor,
    target: torch.Tensor,
    to_bgr: bool = False,
) -> np.ndarray:
    """Generate concatenated frame for validation/benchmark videos
    """
    assert torch.is_tensor(pers) and torch.is_tensor(target)
    _pers = post_process_for_render_torch(pers, to_bgr=to_bgr)
    _target = post_process_for_render_torch(target, to_bgr=to_bgr)
    frame = np.concatenate([_pers, _target], axis=1)
    return frame


def renders_to_image(render: Dict, info: Dict = None) -> np.ndarray:
    """Generate image of single frame from render and info
    returned from a single environment step().
    Args:
        render: observation returned from an environment step().
        info: info returned from an environment step().
    Returns:
        generated image of a single frame.
    """
    pers = render['pers']
    target = render['target']
    render_frame = np.concatenate([pers, target], axis=1)
    return render_frame


def append_text_to_image(image: np.ndarray, text: str):
    """Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
