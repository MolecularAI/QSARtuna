import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
from apischema import serialize, deserialize

from optunaz import optbuild
import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    OptimizationConfig,
    LogisticRegression,
    SVR,
    RandomForest,
    Ridge,
    Lasso,
    PLS,
    XGBregressor,
    SVC,
    AdaBoostClassifier,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, MACCS_keys, ECFP_counts, PhyschemDescriptors


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def optconfig_regression(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForest.new(),
            Ridge.new(),
            Lasso.new(),
            PLS.new(),
            XGBregressor.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=100,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            SVC.new(),
            RandomForest.new(),
            LogisticRegression.new(),
            AdaBoostClassifier.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=3,
            n_trials=100,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_optbuild_regression(optconfig_regression):
    optunaz.three_step_opt_build_merge.optimize(optconfig_regression, "test_regression")


def test_optbuild_classification(optconfig_classification):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_classification, "test_classification"
    )


def test_optbuild_cli(shared_datadir, optconfig_regression):

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_regression)))

    testargs = [
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
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None


def test_physchem_descriptor(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), PhyschemDescriptors.new()],
        algorithms=[RandomForest.new(), XGBregressor.new(),],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=15,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    optunaz.three_step_opt_build_merge.optimize(config, "test_regression_physchem")
