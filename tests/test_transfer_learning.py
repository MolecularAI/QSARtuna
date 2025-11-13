import pytest
from optunaz.config import OptimizationDirection, ModelMode
from optunaz.datareader import Dataset
from optunaz.descriptors import SmilesFromFile

from optunaz.config.optconfig import (
    ChemPropRegressorPretrained,
    OptimizationConfig,
    ChemPropRegressor,
)
from optunaz.three_step_opt_build_merge import optimize, buildconfig_best, build_best


@pytest.mark.skip(reason="Latency")
def test_tl_chemprop(clean_shared_datadir):
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(clean_shared_datadir / "DRD2/subset-50/train.csv"),
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(epochs=1),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    study = optimize(config, study_name="pretrain")
    _ = build_best(buildconfig_best(study), str(clean_shared_datadir / "pretrained.pkl"))

    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(clean_shared_datadir / "DRD2/subset-50/train.csv"),
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressorPretrained.new(
                pretrained_model=str(clean_shared_datadir / "pretrained.pkl"),
                epochs=ChemPropRegressorPretrained.Parameters.ChemPropParametersEpochs(
                    high=4, low=4
                ),
            )
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )

    study = optimize(config, study_name="adapt")
