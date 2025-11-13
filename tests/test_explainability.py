import json
import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from apischema import deserialize, serialize

from optunaz import optbuild, predict
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    AdaBoostClassifier,
    ChemPropClassifier,
    ChemPropRegressor,
    Lasso,
    OptimizationConfig,
    PLSRegression,
    CatBoostRegressor,
    CatBoostClassifier,
    Ridge,
    TabPFNClassifier,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    Avalon,
    CompositeDescriptor,
    ECFP_counts,
    MACCS_keys,
    PhyschemDescriptors,
    SmilesAndSideInfoFromFile,
    SmilesFromFile,
    UnscaledPhyschemDescriptors,
)
from optunaz.utils.preprocessing.splitter import NoSplitting


@pytest.fixture
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_side_info_cls(clean_shared_datadir):
    """Returns 50 molecules and side info from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train_side_info_cls.csv")


@pytest.fixture
def file_sdf1(clean_shared_datadir):
    """Returns sdf test file."""
    return str(clean_shared_datadir / "sdf" / "1.sdf")


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
            n_splits=2,
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
            CatBoostRegressor.new(
                n_estimators=CatBoostRegressor.Parameters.CatboostRegressorParametersNEstimators(
                    low=2, high=2
                )
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="explained_variance",
        ),
    )


@pytest.fixture
def optconfig_regression3(file_drd2_50):
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
            n_splits=2,
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
            n_splits=2,
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
            CatBoostClassifier.new(
                n_estimators=CatBoostClassifier.Parameters.CatboostClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            n_splits=2,
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
            ChemPropRegressor.new(epochs=1),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=1,
            n_replicates=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification_chemprop(file_drd2_50, file_drd2_50_side_info_cls):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            SmilesAndSideInfoFromFile.new(
                file=file_drd2_50_side_info_cls,
                input_column="canonical",
                y_aux_column="ylabels",
            )
        ],
        algorithms=[
            ChemPropClassifier.new(epochs=1),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            n_splits=2,
            n_trials=1,
            n_replicates=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="auc_pr_cal",
        ),
    )


@pytest.fixture
def optconfig_classification_tabpfn(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
            covariate_column="activity",
            split_strategy=NoSplitting()
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            TabPFNClassifier.new(max_time=30, max_feats=2),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            n_splits=2,
            n_trials=1,
            n_replicates=1,
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
    ],
)
def test_qptuna_explainability(file_drd2_50, clean_shared_datadir, optconfig, request):
    optconfig = request.getfixturevalue(optconfig)
    optconfig.set_cache()

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "merged.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
        "--predict-explain",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()
    predictions = pd.read_csv(str(clean_shared_datadir / "outprediction"))
    try:
        assert len(predictions.dropna(subset=["shap_value"])) > 0
    except KeyError:
        assert len(predictions.dropna(subset=["rationale_0_score"])) == 0


def test_interpret_explainability(file_drd2_50, clean_shared_datadir):
    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "DRD2" / "drd2_cls.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
        "--predict-explain",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()
    predictions = pd.read_csv(str(clean_shared_datadir / "outprediction"))
    try:
        assert len(predictions.dropna(subset=["shap_value"])) > 0
    except KeyError:
        assert len(predictions.dropna(subset=["prediction"])) == 50


def test_tabpfn_explainability(
    file_drd2_50,
    optconfig_classification_tabpfn,
    clean_shared_datadir,
):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_classification_tabpfn)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
        "--input-covariate-column",
        "activity",
        "--predict-explain",
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(clean_shared_datadir / "outprediction"), usecols=["shap_value"]
    )
    assert len(predictions.dropna() == 2)
