import os
from copy import deepcopy

import optuna
from apischema import deserialize

from optunaz.config.optconfig import OptimizationConfig
from optunaz.objective import Objective
from optunaz.utils import files_paths, load_json
from optunaz.utils.enums.optimization_configuration_enum import OptimizationConfigurationEnum

import pytest

_OC = OptimizationConfigurationEnum()


@pytest.fixture
def conf(shared_datadir):
    conf = load_json.loadJSON(
        path=os.path.join(
            files_paths.move_up_directory(__file__, 1),
            "examples", "optimization", "RF_SVR_regression.json"
        )
    )
    conf["data"]["training_dataset_file"] = str(shared_datadir / "DRD2/subset-50/train.csv")
    conf["data"]["test_dataset_file"] = str(shared_datadir / "DRD2/subset-50/test.csv")

    return conf


def test_ECFP(conf):
    # overwrite descriptor specification
    confDict = deepcopy(conf)
    confDict[_OC.DESCRIPTORS] = [
        { "name": _OC.DESCRIPTORS_ECFP,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_ECFP_RADIUS: 3,
                _OC.DESCRIPTORS_ECFP_NBITS: 2048
            }
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_ECFP_count(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        { "name": _OC.DESCRIPTORS_ECFPCOUNTS,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_ECFPCOUNTS_RADIUS: 3,
               _OC.DESCRIPTORS_ECFPCOUNTS_USEFEATURES: True
            }
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)

def test_Avalon(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        { "name": _OC.DESCRIPTORS_AVALON,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_AVALON_NBITS: 2048
            }
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)

def test_MACCS_keys(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_MACCSKEYS,
            _OC.GENERAL_PARAMETERS: {}
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)
