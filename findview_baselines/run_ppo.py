#!/usr/bin/env python3

import argparse

from LookAround.config import Config, DictAction

from findview_baselines.rl.ppo.trainer import PPOTrainer
from findview_baselines.rl.ppo.validator import PPOValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml config info about experiment",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=('train', 'eval'),
        default='train',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='arguments in dict',
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    mode: str,
    config: str,
    options=None,
) -> None:
    cfg = Config.fromfile(config)
    if options is not None:
        cfg.merge_from_dict(options)

    print(">>> Config:")
    print(cfg.pretty_text)

    if mode == "train":
        trainer = PPOTrainer(cfg=cfg)
        trainer.train()
    elif mode == "eval":
        validator = PPOValidator(cfg=cfg)
        validator.eval()


if __name__ == "__main__":
    main()
