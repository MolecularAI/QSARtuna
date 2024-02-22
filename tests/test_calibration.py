import pytest

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    RandomForestClassifier,
    CalibratedClassifierCVWithVA,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    MACCS_keys,
    ECFP_counts,
)


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def optconfig_vennabers(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(
                    n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                        low=2, high=2
                    )
                ),
                n_folds=2,
                ensemble="True",
                method="vennabers",
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="neg_brier_score",
        ),
    )


@pytest.fixture
def optconfig_sigmoid(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(
                    n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                        low=2, high=2
                    )
                ),
                n_folds=2,
                ensemble="True",
                method="sigmoid",
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=4,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="auc_pr_cal",
        ),
    )


@pytest.fixture
def optconfig_isotonic(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(
                    n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                        low=2, high=2
                    )
                ),
                n_folds=2,
                ensemble="True",
                method="isotonic",
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=4,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="concordance_index",
        ),
    )


def test_optbuild_va(optconfig_vennabers):
    optunaz.three_step_opt_build_merge.optimize(optconfig_vennabers, "test_vennabers")


def test_optbuild_sigmoid(optconfig_sigmoid):
    optunaz.three_step_opt_build_merge.optimize(optconfig_sigmoid, "test_sigmoid")


def test_optbuild_isotonic(optconfig_isotonic):
    optunaz.three_step_opt_build_merge.optimize(optconfig_isotonic, "test_isotonic")
