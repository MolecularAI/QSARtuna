import argparse
import json
import logging
import os
import pathlib
from typing import Union

from apischema import deserialize, serialize

from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)

from optunaz.config import LOG_CONFIG

log_conf = LOG_CONFIG
logging.config.dictConfig(log_conf)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="optbuild: Optimize hyper-parameters and build (train) the best model."
    )
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to input configuration file (JSON): "
        "either Optimization configuration, "
        "or Build (training) configuration.",
    )
    parser.add_argument(
        "--best-buildconfig-outpath",
        help="Path where to write Json of the best build configuration.",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--best-model-outpath",
        help="Path where to write (persist) the best model.",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--merged-model-outpath",
        help="Path where to write (persist) the model trained on merged train+test data.",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--no-cache",
        help="Turn off descriptor generation caching ",
        action="store_true",
    )
    args = parser.parse_args()

    AnyConfig = Union[OptimizationConfig, BuildConfig]
    with open(args.config, "rt") as fp:
        config = deserialize(AnyConfig, json.load(fp), additional_properties=True)

    if isinstance(config, OptimizationConfig):
        study_name = str(pathlib.Path(args.config).absolute())
        if not args.no_cache:
            config.set_cache()
            cache = config._cache
            cache_dir = config._cache_dir
        else:
            cache = None
            cache_dir = None
        study = optimize(config, study_name=study_name)
        if args.best_model_outpath or args.merged_model_outpath:
            buildconfig = buildconfig_best(study)
    elif isinstance(config, BuildConfig):
        buildconfig = config
        cache = None
        cache_dir = None
    else:
        raise ValueError(f"Unrecognized config type: {type(config)}.")

    if args.best_buildconfig_outpath:
        os.makedirs(os.path.dirname(args.best_buildconfig_outpath), exist_ok=True)
        with open(args.best_buildconfig_outpath, "wt") as fp:
            json.dump(serialize(buildconfig), fp, indent="  ")
    if args.best_model_outpath:
        build_best(
            buildconfig,
            args.best_model_outpath,
            cache=cache,
        )
    if args.merged_model_outpath:
        build_merged(
            buildconfig,
            args.merged_model_outpath,
            cache=cache,
        )
    if cache_dir is not None:
        cache_dir.cleanup()
