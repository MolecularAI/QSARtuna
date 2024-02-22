import pytest

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    PRFClassifier,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, SmilesFromFile
from optunaz.objective import NoValidDescriptors


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
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_threshold=330,
            probabilistic_threshold_representation_std=100,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            PRFClassifier.new(
                n_estimators=PRFClassifier.Parameters.PRFClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=1,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression_noptr_error(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            PRFClassifier.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=1,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_optbuild_regression(optconfig_regression):
    optunaz.three_step_opt_build_merge.optimize(optconfig_regression, "test_regression")


def test_optbuild_regression_ptr_error(optconfig_regression_noptr_error):
    with pytest.raises(
        ValueError,
        match="PRFClassifier supplied but response column outside \[0.0\-1.0\] acceptable range",
    ):
        optunaz.three_step_opt_build_merge.optimize(
            optconfig_regression_noptr_error, "test_regression_ptr_error"
        )


def test_no_compatible_descriptors(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_threshold=330,
            probabilistic_threshold_representation_std=100,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[PRFClassifier.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=1,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            split_chemprop=False,
        ),
    )
    with pytest.raises(
        NoValidDescriptors,
        match="None of the supplied descriptors: \['SmilesFromFile'\] are "
        "compatible with the supplied algo: PRFClassifier.",
    ):
        optunaz.three_step_opt_build_merge.optimize(config, "test_incompatible_smiles")


def test_no_compatible_subspace_nosplit_chemprop(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_threshold=330,
            probabilistic_threshold_representation_std=100,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            PRFClassifier.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=1,
            n_trials=1,
            split_chemprop=False,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    with pytest.raises(
        NoValidDescriptors,
        match="None of the supplied descriptors: \['SmilesFromFile'\] are "
        "compatible with the supplied algo: PRFClassifier.",
    ):
        optunaz.three_step_opt_build_merge.optimize(
            config, "test_incompatible_subspace1"
        )


def test_no_compatible_subspace_split_chemprop(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_threshold=330,
            probabilistic_threshold_representation_std=100,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            PRFClassifier.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=1,
            n_trials=1,
            split_chemprop=True,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    with pytest.raises(
        UnboundLocalError, match="No valid subspaces were found, check your config"
    ):
        optunaz.three_step_opt_build_merge.optimize(
            config, "test_incompatible_subspace2"
        )
