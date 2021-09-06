#!/usr/bin/env python3

import argparse

from mycv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='arguments in dict',
    )
    parser.add_argument(
        '--dump',
        action='store_true',
        help='dump the config as `example.py`',
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')

    if args.dump:
        # dump config
        cfg.dump('configs/example.py')


if __name__ == '__main__':
    main()
