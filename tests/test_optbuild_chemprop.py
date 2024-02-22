import pytest
import tempfile


from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    ChemPropClassifier,
    ChemPropRegressor,
    ChemPropHyperoptRegressor,
    ChemPropHyperoptClassifier,
    XGBRegressor,
)

from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)

from optunaz.datareader import Dataset
from optunaz.descriptors import SmilesFromFile, ECFP, SmilesAndSideInfoFromFile
from optunaz.objective import NoValidDescriptors


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_side_info(shared_datadir):
    """Returns 50 molecules and side info from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train_side_info.csv")


@pytest.fixture
def optconfig_regression(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropHyperoptRegressor.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification(file_drd2_50, file_drd2_50_side_info):
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
        ),
    )


@pytest.fixture
def optconfig_classification_X_aux(file_drd2_50, file_drd2_50_side_info):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
            aux_column="activity",
        ),
        descriptors=[
            SmilesAndSideInfoFromFile.new(
                file=file_drd2_50_side_info,
                input_column="canonical",
            )
        ],
        algorithms=[
            ChemPropClassifier.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification_X_aux_noSI(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
            aux_column="activity",
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropClassifier.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_optbuild_regression(optconfig_regression):
    optimize(optconfig_regression, "test_regression")


def test_optbuild_classification(optconfig_classification):
    optimize(optconfig_classification, "test_classification")


def test_optbuild_classification_Xaux(optconfig_classification_X_aux):
    optimize(optconfig_classification_X_aux, "test_xaux_classification")


def test_optbuild_classification_Xaux_noSI(optconfig_classification_X_aux_noSI):
    optimize(optconfig_classification_X_aux_noSI, "test_xaux_nosi_classification")


def test_chemprop_tolerate_ECFP(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), SmilesFromFile.new()],
        algorithms=[
            ChemPropHyperoptRegressor.new(epochs=4),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            n_chemprop_trials=3,
            direction=OptimizationDirection.MAXIMIZATION,
            random_seed=1,  # seed 1 ensures 2nd trial picks ChemProp & ECFP
            split_chemprop=False,
        ),
    )

    optimize(config, "test_tolerate_ecfp")


def test_no_compatible_descriptors(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[ChemPropRegressor.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=100,
            direction=OptimizationDirection.MAXIMIZATION,
            split_chemprop=False,
        ),
    )

    with pytest.raises(
        NoValidDescriptors,
        match="None of the supplied descriptors: \['ECFP'\] are "
        "compatible with the supplied algo: ChemPropRegressor.",
    ):
        optimize(config, "test_incompatible_ecfp")


def test_chemprop_with_XGB(file_drd2_50):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), SmilesFromFile.new()],
        algorithms=[ChemPropHyperoptRegressor.new(epochs=4), XGBRegressor.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=3,
            n_chemprop_trials=1,
            random_seed=2,
            split_chemprop=False,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    # stage 1
    stage1 = optimize(config, "test_chemprop_with_xgb")
    buildconfig = buildconfig_best(stage1)
    with tempfile.NamedTemporaryFile() as f:
        # stage 2
        build_best(buildconfig, f.name)
    with tempfile.NamedTemporaryFile() as f:
        # stage 3
        build_merged(buildconfig, f.name)
