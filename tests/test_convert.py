import json
import os
import dill
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from apischema import serialize

from optunaz import convert, optbuild, predict
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    CustomRegressionModel,
    KNeighborsClassifier,
    Lasso,
    OptimizationConfig,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    AnyUnscaledDescriptor,
    PrecomputedDescriptorFromFile,
    SmilesFromFile,
    UnscaledZScalesDescriptors,
    MordredDescriptors
)
from optunaz.model_writer import Predictor
from optunaz.three_step_opt_build_merge import optimize


@dataclass
class RegModel(Predictor):
    def predict(self, xs):
        return np.sum(xs, axis=1) > np.mean(xs)

    def fit(self, X, y):
        pass

    def predict_uncert(self, xs):
        return np.sum(xs, axis=1) > np.mean(xs)


@dataclass
class RegMedianModel(Predictor):
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.median_ = np.median(self.y_)
        pass

    def predict(self, xs):
        # Return an array of predicted values, each being the median
        return np.full(shape=(xs.shape[0],), fill_value=self.median_)


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
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def train_with_fp(clean_shared_datadir):
    """Returns sdf test file."""
    return str(clean_shared_datadir / "precomputed_descriptor" / "train_with_fp.csv")


@pytest.fixture
def inference_uncert(clean_shared_datadir):
    """Returns inference_uncert test file."""
    return str(clean_shared_datadir / "peptide" / "permeability" / "train.csv")


def test_reg_convert(clean_shared_datadir):
    with open(str(clean_shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(RegModel(), model_f)

    convert.convert(
        pathlib.Path(clean_shared_datadir / "pkl.pkl"),
        ModelMode.REGRESSION,
        pathlib.Path(str(clean_shared_datadir / "converted.pkl")),
        pathlib.Path(clean_shared_datadir / "ecfp.json"),
    )


def test_cls_convert(clean_shared_datadir):
    with open(str(clean_shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(ClsModel(), model_f)

    convert.convert(
        pathlib.Path(clean_shared_datadir / "pkl.pkl"),
        ModelMode.CLASSIFICATION,
        pathlib.Path(str(clean_shared_datadir / "converted.pkl")),
        pathlib.Path(clean_shared_datadir / "ecfp.json"),
    )


def test_no_predict_err_convert(clean_shared_datadir):
    with open(str(clean_shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(ECFP.new()), ecfp_f)
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(NoPredictModel(), model_f)

    with pytest.raises(
        ValueError,
        match="An estimator with a 'predict' method must be supplied",
    ):
        convert.convert(
            pathlib.Path(clean_shared_datadir / "pkl.pkl"),
            ModelMode.REGRESSION,
            pathlib.Path(clean_shared_datadir / "converted.pkl"),
            pathlib.Path(clean_shared_datadir / "ecfp.json"),
        )


def test_no_predictproba_err_convert(clean_shared_datadir):
    with open(str(clean_shared_datadir / "ecfp.json"), "wt") as ecfp_f:
        json.dump(serialize(SmilesFromFile.new()), ecfp_f)
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(NoPredictProbaModel(), model_f)

    with pytest.raises(
        ValueError,
        match="An estimator with a 'predict_proba' method must be supplied.",
    ):
        convert.convert(
            pathlib.Path(clean_shared_datadir / "pkl.pkl"),
            ModelMode.CLASSIFICATION,
            pathlib.Path(str(clean_shared_datadir / "converted.pkl")),
            pathlib.Path(clean_shared_datadir / "ecfp.json"),
        )


def test_no_JSON_convert(clean_shared_datadir):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(ClsModel(), model_f)

    convert.convert(
        pathlib.Path(clean_shared_datadir / "pkl.pkl"),
        ModelMode.CLASSIFICATION,
        pathlib.Path(str(clean_shared_datadir / "converted.pkl")),
        None,
    )


@pytest.mark.parametrize("descriptor", AnyUnscaledDescriptor.__args__)
def test_cli_predict_convert(
    clean_shared_datadir, file_drd2_50, train_with_fp, descriptor
):
    if isinstance(
        descriptor.new(),
        Union[
            PrecomputedDescriptorFromFile,
            UnscaledZScalesDescriptors,
            MordredDescriptors,
        ].__args__,
    ):
        pytest.skip()
    with open(str(clean_shared_datadir / "descriptor.json"), "wt") as ecfp_f:
        json.dump(serialize(descriptor.new()), ecfp_f)
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
       dill.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--input-json-descriptor-file",
        str(clean_shared_datadir / "descriptor.json"),
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "converted.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(clean_shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


def test_cli_customdesc_predict_convert(
    clean_shared_datadir, file_drd2_50, train_with_fp
):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
    ]
    with patch.object(sys, "argv", convert_args):
        convert.main()

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "converted.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--input-covariate-column",
        "activity",
        "--input-precomputed-file",
        str(train_with_fp),
        "--input-precomputed-input-column",
        "canonical",
        "--input-precomputed-response-column",
        "fp",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(clean_shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


def test_cli_convert_optimise(clean_shared_datadir, file_drd2_50):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
       dill.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
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
                model_file=str(clean_shared_datadir / "converted.pkl")
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    optimize(config, "test_converted")


def test_cli_convert_optimise_refit(clean_shared_datadir, file_drd2_50):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
       dill.dump(RefitCheck(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
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
                model_file=str(clean_shared_datadir / "converted.pkl"),
                refit_model=1,
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    with pytest.raises(
        TypeError,
        match="ValueError: Refit check",
    ):
        optimize(config, "test_refit")


def test_cli_median_predictor(clean_shared_datadir, file_drd2_50):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(RegMedianModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
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
                model_file=str(clean_shared_datadir / "converted.pkl"),
                refit_model=1,
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    optimize(config, "test_refit")


def test_cli_convert_optbuild_al_reg_custom(
    clean_shared_datadir, file_drd2_50, inference_uncert
):
    with open(str(clean_shared_datadir / "pkl.pkl"), "wb") as model_f:
        dill.dump(RegModel(), model_f)

    convert_args = [
        "prog",
        "--input-model-file",
        str(clean_shared_datadir / "pkl.pkl"),
        "--input-model-mode",
        ModelMode.REGRESSION,
        "--output-model-path",
        str(clean_shared_datadir / "converted.pkl"),
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
                model_file=str(clean_shared_datadir / "converted.pkl"),
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(clean_shared_datadir / "al.csv"))
    assert len(predictions.dropna()) == 0

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(clean_shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(clean_shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(clean_shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


def test_cli_convert_optbuild_al_reg_lasso(
    clean_shared_datadir, file_drd2_50, inference_uncert
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
            n_splits=2,
            n_trials=3,
            n_startup_trials=0,
            random_seed=42,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(clean_shared_datadir / "al.csv"))
    print(len(predictions))
    assert len(predictions.dropna(subset=["Prediction", "Prediction_uncert"])) == 17

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(clean_shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(clean_shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(clean_shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


def test_cli_convert_optbuild_al_cls_(
    clean_shared_datadir, file_drd2_50, inference_uncert
):
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
            n_splits=2,
            n_trials=3,
            n_startup_trials=0,
            random_seed=42,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(opt_config)))

    opt_args = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", opt_args):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    build_args = [
        "prog",
        "--config",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "builtconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "built_best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "built_merged.pkl"),
        "--inference_uncert",
        inference_uncert,
    ]
    with patch.object(sys, "argv", build_args):
        optbuild.main()

    predictions = pd.read_csv(str(clean_shared_datadir / "al.csv"))
    assert len(predictions.dropna()) == 25

    for pkl_file in ["best.pkl", "merged.pkl", "built_best.pkl", "built_merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(clean_shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(clean_shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(clean_shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50
