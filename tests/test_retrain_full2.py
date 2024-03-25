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


@pytest.fixture
def public_pim(shared_datadir):
    return str(shared_datadir / "PUBLIC_PIM.csv")

@pytest.fixture
def az_pim(shared_datadir):
    return str(shared_datadir / "PIM1_AZ.csv")

@pytest.fixture
def cp_60(shared_datadir,public_pim):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="P_ACT",
            response_type="regression",
            training_dataset_file=public_pim,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(
                epochs=60,
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=5,
            n_trials=10,
            n_startup_trials=10,
            direction=OptimizationDirection.MAXIMIZATION,
            optuna_storage="sqlite:///" + str(shared_datadir / "optuna_storage1.sqlite")
        ),
    )


@pytest.fixture
def main_optconfig(shared_datadir, public_pim):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="P_ACT",
            response_type="regression",
            training_dataset_file=public_pim,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(
                epochs=60,
            ),
            ChemPropRegressorPretrained.new(
                epochs=ChemPropRegressorPretrained.Parameters.ChemPropParametersEpochs(
                    low=4, high=60, q=1),
                pretrained_model=str(shared_datadir / "pim_60.pkl"),
                frzn=['none','mpnn','mpnn_first_ffn','mpnn_last_ffn']
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=5,
            n_trials=15,
            n_startup_trials=15,
            direction=OptimizationDirection.MAXIMIZATION,
            optuna_storage="sqlite:///" + str(shared_datadir / "optuna_storage1.sqlite")
        ),
    )


def test_retrain_integration(public_pim, shared_datadir, cp_60, main_optconfig):

    for cp, optconfig in [('60',cp_60)]:
        optconfig.set_cache()

        with tempfile.NamedTemporaryFile(
            mode="wt", delete=False, dir=shared_datadir, suffix=".json"
        ) as optconfig_fp:
            optconfig_fp.write(json.dumps(serialize(optconfig)))

        testargs = [
            "prog",
            "--config",
            str(optconfig_fp.name),
            "--best-buildconfig-outpath",
            str(shared_datadir / "buildconfig.json"),
            "--merged-model-outpath",
            str(shared_datadir / f"pim_{cp}.pkl"),
        ]
        with patch.object(sys, "argv", testargs):
            optbuild.main()

        os.unlink(optconfig_fp.name)

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
