#!/usr/bin/env python
#  coding=utf-8

import argparse
import json
import sys
from typing import Union

from apischema import deserialize

from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Validator checking JSON configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="A path to a configuration file (JSON dictionary), that is to be validated.",
    )
    args = parser.parse_args()

    AnyConfig = Union[OptimizationConfig, BuildConfig]
    with open(args.conf, "rt") as fp:
        config = deserialize(AnyConfig, json.load(fp))


if __name__ == "__main__":
    sys.exit(main())
