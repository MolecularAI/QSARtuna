import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Union

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

logger = logging.getLogger(__name__)


def predict_pls(model_path, inference_path):
    if inference_path == "None":
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
        "0.99",
    ]
    try:
        with patch.object(sys, "argv", predict_args):
            logging.info("Performing active learning predictions")
            predict.main()
    except FileNotFoundError:
        logger.info(
            f"PLS file not found at {model_path}, AL predictions not performed"
        )
    except predict.UncertaintyError:
        logging.info(
            "PLS prediction not performed: algorithm does not support uncertainty prediction"
        )
    except predict.AuxCovariateMissing:
        logging.info(
            "PLS prediction not performed: algorithm requires corvariate auxiliary data for inference"
        )


def main():
    logging.basicConfig(level=logging.INFO)
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
        help="Path for uncertainty inference and thresholding.",
        type=pathlib.Path,
        default="/projects/db-mirror/MLDatasets/PLS/pls.csv",
    )
    args = parser.parse_args()

    AnyConfig = Union[OptimizationConfig, BuildConfig]
    with open(args.config, "rt") as fp:
        config = deserialize(AnyConfig, json.load(fp), additional_properties=True)

    if isinstance(config, OptimizationConfig):
        study_name = str(pathlib.Path(args.config).absolute())
        pred_pls = False
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
        pred_pls = True
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
        if not args.merged_model_outpath and pred_pls:
            predict_pls(args.best_model_outpath, args.inference_uncert)
    if args.merged_model_outpath:
        build_merged(
            buildconfig,
            args.merged_model_outpath,
            cache=cache,
        )
        if pred_pls:
            predict_pls(args.merged_model_outpath, args.inference_uncert)
    if cache_dir is not None:
        cache_dir.cleanup()
