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
    RandomForestRegressor,
    Ridge,
    Lasso,
    PLSRegression,
    AdaBoostClassifier,
    OptimizationConfig,
    ChemPropRegressor,
    ChemPropHyperoptClassifier,
)

from optunaz.datareader import Dataset
from optunaz.descriptors import (
    Avalon,
    ECFP,
    MACCS_keys,
    ECFP_counts,
    CompositeDescriptor,
    PhyschemDescriptors,
    UnscaledPhyschemDescriptors,
    SmilesFromFile,
    SmilesAndSideInfoFromFile,
)
from optunaz import predict


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_side_info(shared_datadir):
    """Returns 50 molecules and side info from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train_side_info.csv")


@pytest.fixture
def file_sdf1(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "sdf" / "1.sdf")


@pytest.fixture
def optconfig_regression1(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[MACCS_keys.new()],
        algorithms=[
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="r2",
        ),
    )


@pytest.fixture
def optconfig_regression2(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            CompositeDescriptor.new(
                descriptors=[
                    ECFP.new(),
                    MACCS_keys.new(),
                    UnscaledPhyschemDescriptors.new(),
                ]
            ),
        ],
        algorithms=[
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="explained_variance",
        ),
    )


@pytest.fixture
def optconfig_regression3(file_drd2_50):
    #d = PhyschemDescriptors.new(
    #    rdkit_names=["fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro"]
    #)
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[PhyschemDescriptors.new()],
        algorithms=[
            PLSRegression.new(),
            Lasso.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression4(file_sdf1):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="LogP",
            training_dataset_file=file_sdf1,
        ),
        descriptors=[Avalon.new()],
        algorithms=[
            Ridge.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            AdaBoostClassifier.new(
                n_estimators=AdaBoostClassifier.Parameters.AdaBoostClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="concordance_index",
        ),
    )


@pytest.fixture
def optconfig_regression_chemprop(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification_chemprop(file_drd2_50, file_drd2_50_side_info):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            SmilesAndSideInfoFromFile.new(
                file=file_drd2_50_side_info,
                input_column="canonical",
            )
        ],
        algorithms=[
            ChemPropHyperoptClassifier.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="auc_pr_cal",
        ),
    )


@pytest.mark.parametrize(
    "optconfig",
    [
        "optconfig_classification",
        "optconfig_regression1",
        "optconfig_regression2",
        "optconfig_regression3",
        "optconfig_regression4",
        "optconfig_classification_chemprop",
        "optconfig_regression_chemprop",
    ],
)
def test_qptuna_explainability(file_drd2_50, shared_datadir, optconfig, request):
    optconfig = request.getfixturevalue(optconfig)
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
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
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
        str(shared_datadir / "merged.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
        "--predict-explain",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()
    predictions = pd.read_csv(str(shared_datadir / "outprediction"))
    try:
        assert len(predictions.dropna(subset=["shap_value"])) > 0
    except KeyError:
        assert len(predictions.dropna(subset=["score"])) == 50
