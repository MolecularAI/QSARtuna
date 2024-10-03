import argparse
import json
import logging
import logging.config
import os
import pathlib
import pickle
from apischema import deserialize
from optunaz.config import ModelMode, LOG_CONFIG
from optunaz.descriptors import AnyDescriptor, PrecomputedDescriptorFromFile
from optunaz.model_writer import save_model
from optunaz.config.buildconfig import (
    BuildConfig,
    CustomRegressionModel,
    CustomClassificationModel,
)
from optunaz.model_writer import QSARtunaModel
from optunaz.datareader import Dataset

log_conf = LOG_CONFIG
logging.config.dictConfig(log_conf)
logger = logging.getLogger(__name__)


def prep_build(mode, pkl_estimator, algorithm, descriptor):
    settings = BuildConfig.Settings(mode=mode)
    metadata = BuildConfig.Metadata(
        name=f"ConvertedModel_{pkl_estimator.name}_{descriptor.name}"
    )
    data = Dataset(
        training_dataset_file="nan",
        test_dataset_file=None,
        input_column="nan",
        response_column="nan",
    )
    buildinfo = BuildConfig(
        algorithm=algorithm,
        descriptor=descriptor,
        settings=settings,
        data=data,
        metadata=metadata,
    )
    return buildinfo


def convert(
    pkl_estimator: pathlib.Path,
    mode: ModelMode,
    out_name: pathlib.Path,
    json_descriptor: pathlib.Path | None,
    wrap_for_uncertainty: bool = False,
):
    """Convert a regressor or classifier model and return it."""

    with open(pkl_estimator, "rb") as fid:
        unpickled = pickle.load(fid)
    if isinstance(unpickled, QSARtunaModel):
        if json_descriptor is not None:
            logging.warning(
                "Since input model is from QSARtuna, json_descriptor was supplied but will be ignored"
            )
        descriptor = unpickled.descriptor
    elif json_descriptor is not None:
        with open(json_descriptor, "rt") as fp:
            descriptor = deserialize(
                AnyDescriptor, json.load(fp), additional_properties=True
            )
            logging.info(f"Descriptor {descriptor} will be used from provided JSON")
    else:
        logging.warning(
            "input-json-descriptor-file not provided, default (PrecomputedDescriptorFromFile) will be used"
        )
        descriptor = PrecomputedDescriptorFromFile.new()

    convertedmodel = (
        CustomRegressionModel
        if mode == ModelMode.REGRESSION
        else CustomClassificationModel
    )

    if wrap_for_uncertainty:
        algorithm = convertedmodel.new(
            preexisting_model=str(pkl_estimator), refit_model=1
        )
        buildinfo = prep_build(mode, pkl_estimator, algorithm, descriptor)
        if mode == ModelMode.REGRESSION:
            from optunaz.config.buildconfig import Mapie

            algorithm = Mapie.new(estimator=algorithm, mapie_alpha=0.05)
        else:
            from optunaz.config.buildconfig import CalibratedClassifierCVWithVA

            algorithm = CalibratedClassifierCVWithVA.new(
                estimator=algorithm, n_folds=2, ensemble="True", method="vennabers"
            )
        algorithm = algorithm.estimator()
        algorithm.fit(unpickled.predictor.X_, unpickled.predictor.y_)
        model = save_model(algorithm, buildinfo, out_name, None, None)
    else:
        algorithm = convertedmodel.new(
            preexisting_model=str(pkl_estimator), refit_model=0
        )
        buildinfo = prep_build(mode, pkl_estimator, algorithm, descriptor)
        model = save_model(algorithm.estimator(), buildinfo, out_name, None, None)

    try:
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
    except FileNotFoundError:
        pass
    with open(out_name, "wb") as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an existing sklearn(-like) model into a QSARtuna model"
    )

    # fmt: off
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--input-model-file", type=pathlib.Path, help="Model file name.", required=True)
    requiredNamed.add_argument("--input-model-mode", type=ModelMode, help="Classification or regression mode for the existing model.", required=True)
    requiredNamed.add_argument("--output-model-path", type=pathlib.Path, help="Path where to write the converted model.", required=True)
    parser.add_argument("--input-json-descriptor-file", type=pathlib.Path, help="Name of input JSON file with descriptor configuration. Defaults to PrecomputedDescriptorFromFile", default=None)
    parser.add_argument("--wrap-for-uncertainty", action="store_true", help="Whether to wrap regression in MAPIE or classification in VennAbers Calibrated Classifiers for uncertainty support")

    # fmt: on

    args, leftovers = parser.parse_known_args()

    assert args.output_model_path.suffix == ".pkl", "Output must be a .pkl file"

    convert(
        args.input_model_file,
        args.input_model_mode,
        args.output_model_path,
        args.input_json_descriptor_file,
        args.wrap_for_uncertainty,
    )
