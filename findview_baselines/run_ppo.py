#!/usr/bin/env python3

import argparse

from LookAround.config import Config, DictAction

from findview_baselines.rl.ppo.ppo_trainer import PPOTrainer


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
        choices=['train', 'test'],
        required=True,
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

    trainer = PPOTrainer(cfg=cfg)

    if mode == "train":
        trainer.train()
    elif mode == "test":
        trainer.test()


if __name__ == "__main__":
    main()
