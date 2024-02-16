import json
import os

import optuna
import pytest
from apischema import deserialize
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial

from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import AnyDescriptor
from optunaz.objective import Objective
from optunaz.utils import load_json
from optunaz.utils.enums import TrialParams
from optunaz.utils.files_paths import attach_root_path


def descriptor_from_trial(trial: FrozenTrial) -> AnyDescriptor:
    descr_str = trial.params[TrialParams.DESCRIPTOR]
    descr_dict = json.loads(descr_str)
    descr = deserialize(AnyDescriptor, descr_dict)
    return descr


@pytest.fixture
def conf(shared_datadir):
    conf = load_json.loadJSON(
        path=os.path.join(
            attach_root_path("examples/optimization/RF_SVR_regression_ECFP_MACCS.json"),
        )
    )
    conf["data"]["training_dataset_file"] = str(
        shared_datadir / "DRD2/subset-50/train.csv"
    )
    conf["data"]["test_dataset_file"] = str(shared_datadir / "DRD2/subset-50/test.csv")
    conf["data"]["input_column"] = "canonical"
    conf["data"]["response_column"] = "molwt"

    return conf


def test_multiple_descriptors(conf):
    config = deserialize(OptimizationConfig, conf)
    train_smiles, train_y, _, test_smiles, test_y, _ = config.data.get_sets()
    obj1 = Objective(optconfig=config, train_smiles=train_smiles, train_y=train_y)
    sampler = TPESampler(seed=42)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler, direction=config.settings.direction)
    study.optimize(obj1, n_trials=4)
    descriptors = {descriptor_from_trial(trial).name for trial in study.trials}
    assert "ECFP" in descriptors
    assert "MACCS_keys" in descriptors
