#!/usr/bin/env python3

import argparse
import os

import torch

from mmengine import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert")
    parser.add_argument("ckpt", help="ckpt path")
    args = parser.parse_args()

    ckpt_path = args.ckpt

    assert os.path.exists(ckpt_path)
    d = torch.load(ckpt_path, map_location="cpu")

    d["cfg"] = Config.fromstring(d["cfg"].pretty_text, file_format=".py")
    torch.save(d, ckpt_path)
