import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from typing import Union
from unittest.mock import patch
import numpy as np
import pandas as pd
from apischema import serialize, deserialize
import pathlib
import pytest
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    CustomRegressionModel,
    Lasso,
    KNeighborsClassifier,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    SmilesFromFile,
    AnyUnscaledDescriptor,
    PrecomputedDescriptorFromFile,
    UnscaledZScalesDescriptors,
)
from optunaz.model_writer import Predictor
from optunaz import convert, predict, optbuild
from optunaz.three_step_opt_build_merge import optimize


@dataclass
class RegModel(Predictor):
    def predict(self, xs):
        return np.sum(xs, axis=1) > np.mean(xs)

    def fit(self, X, y):
        pass


@dataclass
class ClsModel(Predictor):
    def predict(self, xs):
        return np.sum(xs, axis=1)

    def predict_proba(self, xs):
        return np.sum(xs)

    def fit(self, X, y):
        pass


@dataclass
class NoPredictModel:
    def predict_proba(self, xs):
        return np.sum(xs)


@dataclass
class NoPredictProbaModel:
    pass


@dataclass
class RefitCheck(Predictor):
    def predict(self, xs):
        return np.sum(xs, axis=1)

    def fit(self, X, y):
        raise ValueError("Refit check")


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def train_with_fp(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "precomputed_descriptor" / "train_with_fp.csv")


@pytest.fixture
def inference_uncert(shared_datadir):
    """Returns inference_uncert test file."""
    return str(shared_datadir / "peptide" / "permeability" / "train.csv")


def test_reg_convert(shared_datadir):
    with open(str(shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RegModel(), model_f)

    convert.convert(
        pathlib.Path(shared_datadir / "pkl.pkl"),
        ModelMode.REGRESSION,
        pathlib.Path(str(shared_datadir / "converted.pkl")),
        pathlib.Path(shared_datadir / "ecfp.json"),
    )


def test_cls_convert(shared_datadir):
    with open(str(shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(ClsModel(), model_f)

    convert.convert(
        pathlib.Path(shared_datadir / "pkl.pkl"),
        ModelMode.CLASSIFICATION,
        pathlib.Path(str(shared_datadir / "converted.pkl")),
        pathlib.Path(shared_datadir / "ecfp.json"),
    )


def test_no_predict_err_convert(shared_datadir):
    with open(str(shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(NoPredictModel(), model_f)

    with pytest.raises(
        AssertionError,
        match="an estimator with 'predict' method must be supplied to CustomRegressionModel",
    ):
        convert.convert(
            pathlib.Path(shared_datadir / "pkl.pkl"),
            ModelMode.REGRESSION,
            pathlib.Path(shared_datadir / "converted.pkl"),
            pathlib.Path(shared_datadir / "ecfp.json"),
        )


def test_no_predictproba_err_convert(shared_datadir):
    with open(str(shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(SmilesFromFile.new()), ecfp_f)
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(NoPredictProbaModel(), model_f)

    with pytest.raises(
        AssertionError,
        match="an estimator with 'predict_proba' method must be supplied to CustomClassificationModel",
    ):
        convert.convert(
            pathlib.Path(shared_datadir / "pkl.pkl"),
            ModelMode.CLASSIFICATION,
            pathlib.Path(str(shared_datadir / "converted.pkl")),
            pathlib.Path(shared_datadir / "ecfp.json"),
        )


def test_no_JSON_convert(shared_datadir):
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(ClsModel(), model_f)

    convert.convert(
        pathlib.Path(shared_datadir / "pkl.pkl"),
        ModelMode.CLASSIFICATION,
        pathlib.Path(str(shared_datadir / "converted.pkl")),
        None,
    )


@pytest.mark.parametrize("descriptor", AnyUnscaledDescriptor.__args__)
def test_cli_predict_convert(shared_datadir, file_drd2_50, train_with_fp, descriptor):
    if isinstance(
        descriptor.new(),
        Union[PrecomputedDescriptorFromFile, UnscaledZScalesDescriptors].__args__,
    ):
        pytest.skip()
    with open(str(shared_datadir / "descriptor.json"), "wt") as ecfp_f:
        json.dump(serialize(descriptor.new()), ecfp_f)
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--input-json-descriptor-file",
        str(shared_datadir / "descriptor.json"),
        "--output-model-path",
        str(shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "converted.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


def test_cli_customdesc_predict_convert(shared_datadir, file_drd2_50, train_with_fp):
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "converted.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--input-aux-column",
        "activity",
        "--input-precomputed-file",
        str(train_with_fp),
        "--input-precomputed-input-column",
        "canonical",
        "--input-precomputed-response-column",
        "fp",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


def test_cli_convert_optimise(shared_datadir, file_drd2_50):
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            CustomRegressionModel.new(
                preexisting_model=str(shared_datadir / "converted.pkl")
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    optimize(config, "test_converted")


def test_cli_convert_optimise_refit(shared_datadir, file_drd2_50):
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RefitCheck(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            CustomRegressionModel.new(
                preexisting_model=str(shared_datadir / "converted.pkl"),
                refit_model=1,
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    with pytest.raises(
        TypeError,
        match="ValueError: Refit check",
    ):
        optimize(config, "test_refit")


def test_cli_convert_optbuild_al_reg_custom(
    shared_datadir, file_drd2_50, inference_uncert
):
    with open(str(shared_datadir / "pkl.pkl"), "wb") as model_f:
        pickle.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    opt_config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            CustomRegressionModel.new(
                preexisting_model=str(shared_datadir / "converted.pkl"),
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(shared_datadir / "al.csv"))
    assert len(predictions.dropna()) == 0

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


def test_cli_convert_optbuild_al_reg_lasso(
    shared_datadir, file_drd2_50, inference_uncert
):
    opt_config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[Lasso.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            n_startup_trials=0,
            random_seed=42,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(shared_datadir / "al.csv"))
    print(len(predictions))
    assert len(predictions.dropna()) == 20

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


def test_cli_convert_optbuild_al_cls_(shared_datadir, file_drd2_50, inference_uncert):
    opt_config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[KNeighborsClassifier.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=3,
            n_startup_trials=0,
            random_seed=42,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(shared_datadir / "al.csv"))
    assert len(predictions.dropna()) == 25

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50
