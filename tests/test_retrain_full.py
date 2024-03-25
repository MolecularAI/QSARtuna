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
    OptimizationConfig,
    ChemPropRegressor,
    ChemPropRegressorPretrained
)
from optunaz.datareader import Dataset
from optunaz.descriptors import SmilesFromFile
from optunaz import predict
from optunaz.utils.preprocessing.splitter import NoSplitting
from optunaz.utils.preprocessing.deduplicator import KeepMedian

@pytest.fixture
def az_pim(shared_datadir):
    return str(shared_datadir / "PIM1_AZ.csv")


@pytest.fixture
def main_optconfig(shared_datadir, az_pim):
    return OptimizationConfig(
        data=Dataset(
            input_column="Structure",
            response_column="gmean_pIC50",
            response_type="regression",
            training_dataset_file=az_pim,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(
                epochs=30,
            ),
            ChemPropRegressorPretrained.new(
                epochs=ChemPropRegressorPretrained.Parameters.ChemPropParametersEpochs(
                    low=4, high=30, q=1),
                pretrained_model=str(shared_datadir / "pim_60.pkl"),
                frzn=['none','mpnn','mpnn_first_ffn','mpnn_last_ffn']
            ),
            ChemPropRegressorPretrained.new(
                epochs=ChemPropRegressorPretrained.Parameters.ChemPropParametersEpochs(
                    low=0, high=0, q=1),
                pretrained_model=str(shared_datadir / "pim_60.pkl"),
                frzn=['none']
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=5,
            n_trials=15,
            n_startup_trials=15,
            direction=OptimizationDirection.MAXIMIZATION,
            optuna_storage="sqlite:///" + str(shared_datadir / "optuna_storage.sqlite")
        ),
    )


def test_retrain_integration(az_pim, shared_datadir, main_optconfig):

    main_optconfig.set_cache()

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(main_optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "pim_best2.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "pim_merged2.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None
