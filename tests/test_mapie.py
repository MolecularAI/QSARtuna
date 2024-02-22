import pytest

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    RandomForestRegressor,
    XGBRegressor,
    PRFClassifier,
    SVR,
    Mapie,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, PhyschemDescriptors, MACCS_keys


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_sdf1(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "sdf" / "1.sdf")


def optconfig_mapie(file_drd2_50, estimator, descriptor):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[descriptor],
        algorithms=[Mapie.new(estimator=estimator)],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def optconfig_mapie_sdf(file_sdf, estimator, descriptor):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="LogP",
            response_type="regression",
            training_dataset_file=file_sdf,
        ),
        descriptors=[descriptor],
        algorithms=[Mapie.new(estimator=estimator)],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize(
    "test_file,opt,estimator,descriptor",
    [
        (
            "file_drd2_50",
            optconfig_mapie,
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
            ECFP.new(),
        ),
        (
            "file_sdf1",
            optconfig_mapie_sdf,
            XGBRegressor.new(),
            PhyschemDescriptors.new(),
        ),
        ("file_drd2_50", optconfig_mapie, SVR.new(), MACCS_keys.new()),
    ],
)
def test_mapie_estimators(
    test_file, opt, estimator, descriptor, request, shared_datadir
):
    test_file = request.getfixturevalue(test_file)
    optconfig = opt(test_file, estimator, descriptor)
    optconfig.set_cache()
    optunaz.three_step_opt_build_merge.optimize(
        optconfig, f"test_mapie_{test_file}_{estimator}_{descriptor}"
    )


@pytest.fixture
def optconfig_mapie_prfc(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_std=50,
            probabilistic_threshold_representation_threshold=330,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            Mapie.new(
                estimator=PRFClassifier.new(
                    n_estimators=PRFClassifier.Parameters.PRFClassifierParametersNEstimators(
                        low=5, high=5
                    )
                ),
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=5,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_mapie_prf(file_drd2_50, shared_datadir, optconfig_mapie_prfc):
    optunaz.three_step_opt_build_merge.optimize(optconfig_mapie_prfc, f"test_mapie_prf")
