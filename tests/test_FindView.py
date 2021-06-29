#!/usr/bin/env python3


DATASET = [
    dict(
        img_path="indoor/bedroom/pano_azwlwvnimluvvt.jpg",
        initial_view=(0, 0, 0),
        target_view=(0, 45, 45),
    ),
    dict(
        img_path="indoor/bedroom/pano_azwlwvnimluvvt.jpg",
        initial_view=(0, 45, 45),
        target_view=(0, 0, 0),
    ),
]


def test_sample():
    pass
