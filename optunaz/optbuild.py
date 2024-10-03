import argparse
import json
import logging
import logging.config
import os
import pathlib
import sys
from typing import Union
import time

from apischema import deserialize, serialize

from optunaz import predict
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from unittest.mock import patch

from optunaz.config import LOG_CONFIG

log_conf = LOG_CONFIG
logging.config.dictConfig(log_conf)
logger = logging.getLogger(__name__)


def build_with_al(model_path, inference_path, mode):
    """Active learning inference which can occur with buiding"""
    if not inference_path:
        logger.info(f"Inference path is not set so AL predictions not performed")
        return
    else:
        logger.info(f"Inference path is {inference_path}")
    predict_args = [
        "prog",
        "--model-file",
        str(model_path),
        "--input-smiles-csv-file",
        str(inference_path),
        "--input-smiles-csv-column",
        "Structure",
        "--output-prediction-csv-file",
        str(os.path.dirname(model_path)) + "/al.csv",
        "--predict-uncertainty",
        "--uncertainty_quantile",
        "0.50",
    ]
    try:
        with patch.object(sys, "argv", predict_args):
            logging.info("Performing active learning predictions")
            predict.main()
    except FileNotFoundError as e:
        logger.info(
            f"File not found during active learning (AL) prediction, AL predictions not performed: {e}"
        )
    except predict.UncertaintyError:
        logging.info(
            "Uncertainty prediction not supported by algorithm, a temporary uncertainty compatible version will be generated"
        )
        from optunaz import convert

        convert_args = [
            "prog",
            "--input-model-file",
            str(model_path),
            "--input-model-mode",
            mode,
            "--output-model-path",
            str(os.path.dirname(model_path)) + "/al_model.pkl",
            "--wrap-for-uncertainty",
        ]
        with patch.object(sys, "argv", convert_args):
            convert.main()

        build_with_al(
            str(os.path.dirname(model_path)) + "/al_model.pkl", inference_path, mode
        )

    except predict.AuxCovariateMissing:
        logging.info(
            "Active learning (AL) prediction not performed: algorithm requires corvariate auxiliary data for inference"
        )


def basename_from_config(nm: str) -> (pathlib.Path, str):
    """Basename for automatic naming purposes"""
    p, n = os.path.split(nm)
    b, e = os.path.splitext(n)
    base = b[:]
    for repl in ["config_", "conf_"]:
        base = base.replace(repl, "")
        base = base.replace(repl.upper(), "")
    return pathlib.Path(p).absolute(), base


def set_default_output_names(args) -> Union[str, bool]:
    """Set default output names based on the conf file name, where not supplied"""

    if not args.config:
        return False

    if not os.path.exists(args.config):
        logger.critical(
            f"\n!!! Please specify a valid configuration file, {args.config} does not exist !!!\n"
        )
        return False

    basepath, basename = basename_from_config(args.config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if args.best_buildconfig_outpath is None:
        args.best_buildconfig_outpath = pathlib.Path(
            os.path.join(basepath, f"{timestamp}_model_{basename}_best.json")
        )
        logger.info(
            f"best-buildconfig-outpath: set to {args.best_buildconfig_outpath} based on config file name"
        )
    if args.best_model_outpath is None:
        args.best_model_outpath = pathlib.Path(
            os.path.join(basepath, f"{timestamp}_model_{basename}_best.pkl")
        )
        logger.info(
            f"best-model-outpath: set to {args.best_model_outpath} based on config file name"
        )
    if args.merged_model_outpath is None:
        args.merged_model_outpath = pathlib.Path(
            os.path.join(basepath, f"{timestamp}_model_{basename}_final.pkl")
        )
        logger.info(
            f"merged-model-outpath: set to {args.merged_model_outpath} based on config file name"
        )

    return args


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
    parser.add_argument(
        "--inference_uncert",
        help="Path for active learning (AL) predictions to be generated (will activate AL during build).",
        type=pathlib.Path,
        default=None,
    )
    args = parser.parse_args()
    args = set_default_output_names(args)

    AnyConfig = Union[OptimizationConfig, BuildConfig]
    with open(args.config, "rt") as fp:
        config = deserialize(AnyConfig, json.load(fp), additional_properties=True)

    if isinstance(config, OptimizationConfig):
        study_name = str(pathlib.Path(args.config).absolute())
        build_al = False
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
        build_al = True
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
        if not args.merged_model_outpath and build_al:
            build_with_al(
                args.best_model_outpath,
                args.inference_uncert,
                buildconfig.settings.mode,
            )
    if args.merged_model_outpath:
        build_merged(
            buildconfig,
            args.merged_model_outpath,
            cache=cache,
        )
        if build_al:
            build_with_al(
                args.merged_model_outpath,
                args.inference_uncert,
                buildconfig.settings.mode,
            )
    if cache_dir is not None:
        cache_dir.cleanup()
