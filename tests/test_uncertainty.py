import json
import os
import sys
import tempfile
from unittest.mock import patch
import pandas as pd


import pytest
from apischema import serialize, deserialize

from optunaz import optbuild
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    RandomForestClassifier,
    OptimizationConfig,
    ChemPropHyperoptClassifier,
    CalibratedClassifierCVWithVA,
    Mapie,
    ChemPropHyperoptRegressor,
    ChemPropRegressor,
    RandomForestRegressor,
)

from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    SmilesFromFile,
)
from optunaz import predict
from optunaz.utils.preprocessing.transform import LogBase, LogNegative


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_err(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train_with_errors.csv")


@pytest.fixture
def pretrained_cp(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "drd2_reg.pkl")


def optconfig_CalibratedClassifierCVWithVA(file_drd2_50, method, estimator, descriptor):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[descriptor],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=estimator,
                ensemble="True",
                method=method,
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            n_startup_trials=0,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize("method", ["sigmoid", "vennabers"])
@pytest.mark.parametrize(
    "estimator,descriptor",
    [
        (
            ChemPropHyperoptClassifier.new(epochs=4),
            SmilesFromFile.new(),
        ),
        (
            RandomForestClassifier.new(
                n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
            ECFP.new(),
        ),
    ],
)
def test_CalibratedClassifierCVWithVA_uncertainty(
    file_drd2_50,
    shared_datadir,
    method,
    estimator,
    descriptor,
):
    optconfig = optconfig_CalibratedClassifierCVWithVA(
        file_drd2_50, method, estimator, descriptor
    )
    optconfig.set_cache()

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
        "--predict-uncertainty",
    ]
    with patch.object(sys, "argv", predict_args):
        try:
            predict.main()
        except AttributeError:
            # only "isotonic" or "sigmoid" return an attribute error
            assert method in ["isotonic", "sigmoid"]
            return
        else:
            # assert vennabers is the only method with uncertainty
            assert method == "vennabers"

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction_uncert"]
    )
    assert len(predictions.dropna()) == 50


def optconfig_mapie(file_drd2_50, estimator, descriptor):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[descriptor],
        algorithms=[
            Mapie.new(
                estimator=estimator,
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            n_startup_trials=0,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize(
    "estimator,descriptor",
    [
        (
            RandomForestRegressor.new(
                n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
            ECFP.new(),
        ),
    ],
)
def test_mapie_uncertainty(
    file_drd2_50,
    shared_datadir,
    estimator,
    descriptor,
):
    optconfig = optconfig_mapie(file_drd2_50, estimator, descriptor)
    optconfig.set_cache()

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
        "--predict-uncertainty",
        "--uncertainty_quantile",
        "0.55",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 23


def optconfig_chemprop(file_drd2_50, estimator, descriptor):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            log_transform=True,
            log_transform_base=LogBase.LOG10,
            log_transform_negative=LogNegative.TRUE,
            log_transform_unit_conversion=2,
        ),
        descriptors=[descriptor],
        algorithms=[
            estimator,
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            n_startup_trials=0,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize(
    "estimator,descriptor",
    [
        (
            ChemPropRegressor.new(
                epochs=4,
                ensemble_size=4,
            ),
            SmilesFromFile.new(),
        ),
    ],
)
def test_cp_uncertainty(
    file_drd2_50,
    file_drd2_50_err,
    shared_datadir,
    estimator,
    descriptor,
):
    optconfig = optconfig_chemprop(file_drd2_50, estimator, descriptor)

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50_err,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
        "--predict-uncertainty",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction_uncert"]
    )
    assert len(predictions.dropna()) == 47


def test_pretrained_uncertainty(
    file_drd2_50, file_drd2_50_err, shared_datadir, pretrained_cp
):
    predict_args = [
        "prog",
        "--model-file",
        str(pretrained_cp),
        "--input-smiles-csv-file",
        file_drd2_50_err,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
        "--predict-uncertainty",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction_uncert"]
    )
    assert len(predictions.dropna()) == 47
