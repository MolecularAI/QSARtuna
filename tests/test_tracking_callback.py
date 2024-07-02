import pytest

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    Ridge,
    Lasso,
    PLSRegression,
    RandomForestRegressor,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    MACCS_keys,
    ECFP_counts,
    JazzyDescriptors,
    PhyschemDescriptors,
)


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
        descriptors=[
            ECFP.new(),
            ECFP_counts.new(),
            MACCS_keys.new(),
        ],
        algorithms=[
            RandomForestRegressor.new(),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=30,
            n_startup_trials=10,
            direction=OptimizationDirection.MAXIMIZATION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


@pytest.fixture
def optconfig_physchem(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[PhyschemDescriptors.new()],
        algorithms=[RandomForestRegressor.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_trials=1,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


@pytest.fixture
def optconfig_stringentjazzy(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            ECFP.new(),
            JazzyDescriptors.new(
                jazzy_filters={"NumHAcceptors": -1, "NumHDonors": -1, "MolWt": 0}
            ),
        ],
        algorithms=[RandomForestRegressor.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_trials=1,
            random_seed=42,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


def test_optbuild_regression(optconfig_regression):
    optunaz.three_step_opt_build_merge.optimize(optconfig_regression, "test_regression")


def test_physchem(optconfig_physchem):
    optunaz.three_step_opt_build_merge.optimize(optconfig_physchem, "test_physchem")


def test_optbuild_stringent(optconfig_stringentjazzy):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_stringentjazzy, "test_stringent"
    )
