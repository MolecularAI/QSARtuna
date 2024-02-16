import pytest

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    LogisticRegression,
    SVR,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    Lasso,
    PLSRegression,
    XGBRegressor,
    SVC,
    AdaBoostClassifier,
    CalibratedClassifierCVWithVA,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    MACCS_keys,
    ECFP_counts,
    PhyschemDescriptors,
    SmilesFromFile,
)
from optunaz.objective import NoValidDescriptors


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_sdf1(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "sdf" / "1.sdf")


@pytest.fixture
def optconfig_stdev_reg(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[SVR.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
            minimise_std_dev=True,
        ),
    )


@pytest.fixture
def optconfig_stdev_cls(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[SVC.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
            minimise_std_dev=True,
        ),
    )


@pytest.fixture
def optconfig_regression(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=4, high=4
                )
            ),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression_sdf(file_sdf1):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="LogP",
            training_dataset_file=file_sdf1,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            Ridge.new(),
            Lasso.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=5,
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
            SVC.new(),
            RandomForestClassifier.new(
                n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                    low=4, high=4
                )
            ),
            LogisticRegression.new(),
            AdaBoostClassifier.new(
                n_estimators=AdaBoostClassifier.Parameters.AdaBoostClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_optbuild_minimise_stdev_reg(optconfig_stdev_reg):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_stdev_reg, "test_stddev_regression"
    )


def test_optbuild_minimise_stdev_cls(optconfig_stdev_cls):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_stdev_cls, "test_stddev_classification"
    )


def test_optbuild_regression(optconfig_regression):
    optunaz.three_step_opt_build_merge.optimize(optconfig_regression, "test_regression")


def test_optbuild_regression_sdf(optconfig_regression_sdf):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_regression_sdf, "test_regression_sdf"
    )


def test_optbuild_classification(optconfig_classification):
    optunaz.three_step_opt_build_merge.optimize(
        optconfig_classification, "test_classification"
    )


def test_physchem_descriptor(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), PhyschemDescriptors.new()],
        algorithms=[
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=4, high=4
                )
            ),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    config.set_cache()
    optunaz.three_step_opt_build_merge.optimize(config, "test_regression_physchem")


def test_SmilesFromFile_incompatability(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), SmilesFromFile.new()],
        algorithms=[
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=4, high=4
                )
            ),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
            random_seed=1,  # ensures SmilesFromFile in trials
            split_chemprop=False,
        ),
    )
    optunaz.three_step_opt_build_merge.optimize(config, "test_incompatible_smiles")


def test_no_compatible_descriptors(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=5, high=5
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
            split_chemprop=False,
        ),
    )

    with pytest.raises(
        NoValidDescriptors,
        match="None of the supplied descriptors: \['SmilesFromFile'\] are "
        "compatible with the supplied algo: RandomForest",
    ):
        optunaz.three_step_opt_build_merge.optimize(config, "test_incompatible_smiles")


def test_optconfig_calibrated(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(
                    n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                        low=5, high=5
                    )
                ),
                n_folds=3,
                ensemble="False",
                method="vennabers",
            ),
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(
                    n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                        low=5, high=5
                    )
                ),
                n_folds=2,
                ensemble="False",
                method="vennabers",
            ),
            CalibratedClassifierCVWithVA.new(
                estimator=LogisticRegression.new(),
                n_folds=2,
                ensemble="False",
                method="sigmoid",
            ),
            RandomForestClassifier.new(
                n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                    low=5, high=5
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=12,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="neg_brier_score",
        ),
    )
    optunaz.three_step_opt_build_merge.optimize(config, "test_calibration")
