import os
from copy import deepcopy

import optuna
from apischema import deserialize

from optunaz.config.optconfig import OptimizationConfig
from optunaz.objective import Objective
from optunaz.utils import files_paths, load_json
from optunaz.utils.enums.optimization_configuration_enum import (
    OptimizationConfigurationEnum,
)

import pytest

_OC = OptimizationConfigurationEnum()


@pytest.fixture
def conf(shared_datadir):
    conf = load_json.loadJSON(
        path=os.path.join(
            files_paths.move_up_directory(__file__, 1),
            "examples",
            "optimization",
            "RF_SVR_regression.json",
        )
    )
    conf["data"]["training_dataset_file"] = str(
        shared_datadir / "DRD2/subset-50/train.csv"
    )
    conf["data"]["test_dataset_file"] = str(shared_datadir / "DRD2/subset-50/test.csv")

    return conf


@pytest.fixture
def conf_conc(shared_datadir):
    conf = load_json.loadJSON(
        path=os.path.join(
            files_paths.move_up_directory(__file__, 1),
            "examples",
            "optimization",
            "RF_SVR_regression.json",
        )
    )
    conf["data"]["training_dataset_file"] = str(
        shared_datadir / "aux_descriptors_datasets/train_with_conc.csv"
    )
    conf["data"]["test_dataset_file"] = str(
        shared_datadir / "aux_descriptors_datasets/train_with_conc.csv"
    )

    return conf


@pytest.fixture
def confChemProp(shared_datadir):
    conf = load_json.loadJSON(
        path=os.path.join(
            files_paths.move_up_directory(__file__, 1),
            "examples",
            "optimization",
            "ChemProp_drd2_50.json",
        )
    )
    conf["data"]["training_dataset_file"] = str(
        shared_datadir / "DRD2/subset-50/train.csv"
    )
    conf["data"]["test_dataset_file"] = str(shared_datadir / "DRD2/subset-50/test.csv")

    return conf


@pytest.fixture
def train_with_fp(shared_datadir):
    return str(shared_datadir / "precomputed_descriptor" / "train_with_fp.csv")


@pytest.fixture
def train_with_conc(shared_datadir):
    return str(shared_datadir / "aux_descriptors_datasets" / "train_with_conc.csv")


@pytest.fixture
def train_with_si(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-50" / "train_side_info.csv")


def test_Avalon(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_AVALON,
            _OC.GENERAL_PARAMETERS: {_OC.DESCRIPTORS_AVALON_NBITS: 2048},
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_ECFP(conf):
    # overwrite descriptor specification
    confDict = deepcopy(conf)
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_ECFP,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_ECFP_RADIUS: 3,
                _OC.DESCRIPTORS_ECFP_NBITS: 2048,
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_ECFP_count(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_ECFPCOUNTS,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_ECFPCOUNTS_RADIUS: 3,
                _OC.DESCRIPTORS_ECFPCOUNTS_USEFEATURES: True,
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_MACCS_keys(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {"name": _OC.DESCRIPTORS_MACCSKEYS, _OC.GENERAL_PARAMETERS: {}}
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_PhyschemDescriptors(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_PHYSCHEM,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_PHYSCHEM_RDKITNAMES: [
                    "MaxEStateIndex",
                    "MinEStateIndex",
                ]
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_JazzyDescriptors(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_JAZZY,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_JAZZY_JAZZYNAMES: [
                    "dga",
                    "dgp",
                ]
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    conf.Settings.n_jobs = 1
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_PrecomputedDescriptorFromFile(conf, train_with_fp):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_PRECOMPUTED,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_PRECOMPUTED_FILE: train_with_fp,
                _OC.DESCRIPTORS_PRECOMPUTED_INPUT_COLUMNN: "canonical",
                _OC.DESCRIPTORS_PRECOMPUTED_RESPONSE_COLUMN: "fp",
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_PrecomputedDescriptorFromFileWithAux(conf_conc, train_with_conc):
    # overwrite descriptor specification
    confDict = conf_conc
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_PRECOMPUTED,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_PRECOMPUTED_FILE: train_with_conc,
                _OC.DESCRIPTORS_PRECOMPUTED_INPUT_COLUMNN: "canonical",
                _OC.DESCRIPTORS_PRECOMPUTED_RESPONSE_COLUMN: "aux1",
            },
        }
    ]

    # generate optimization configuration and execute it
    conf_conc = deserialize(OptimizationConfig, confDict)
    conf_conc.data.aux_column = "aux2"
    train_smiles, train_y, _, _, _, _ = conf_conc.data.get_sets()
    obj = Objective(optconfig=conf_conc, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf_conc.settings.direction)
    study.optimize(obj, n_trials=1)


def test_SmilesFromFile(confChemProp):
    # overwrite descriptor specification
    confDict = confChemProp
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_SMILES,
            _OC.GENERAL_PARAMETERS: {},
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_SmilesAndSideInfoFromFile(confChemProp, train_with_si):
    # overwrite descriptor specification
    confDict = confChemProp
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_SMILES_AND_SI,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_SMILES_AND_SI_FILE: train_with_si,
                _OC.DESCRIPTORS_SMILES_AND_SI_INPUT_COLUMN: "canonical",
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_ScaledDescriptor(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_SCALED,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_SCALED_DESCRIPTOR: {
                    "name": _OC.DESCRIPTORS_UNSC_PHYSCHEM,
                    _OC.GENERAL_PARAMETERS: {
                        _OC.DESCRIPTORS_PHYSCHEM_RDKITNAMES: [
                            "MaxEStateIndex",
                            "MinEStateIndex",
                        ]
                    },
                }
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_ScaledDescriptor2(conf):
    # overwrite descriptor specification
    confDict = conf
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_SCALED,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS_SCALED_DESCRIPTOR: {
                    "name": _OC.DESCRIPTORS_UNSC_JAZZY,
                    _OC.GENERAL_PARAMETERS: {
                        _OC.DESCRIPTORS_JAZZY_JAZZYNAMES: [
                            "dga",
                            "dga",
                        ]
                    },
                }
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_CompositeDescriptor(conf):
    # overwrite descriptor specification
    confDict = conf.copy()
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_COMPOSITE,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS: [
                    {"name": _OC.DESCRIPTORS_ECFP, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_PHYSCHEM, _OC.GENERAL_PARAMETERS: {}},
                ]
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    conf.set_cache()
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y, cache=conf._cache)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)


def test_AllDescriptorsAsCompositeDescriptor(conf, train_with_fp):
    # overwrite descriptor specification
    confDict = conf.copy()
    confDict[_OC.DESCRIPTORS] = [
        {
            "name": _OC.DESCRIPTORS_COMPOSITE,
            _OC.GENERAL_PARAMETERS: {
                _OC.DESCRIPTORS: [
                    {"name": _OC.DESCRIPTORS_ECFP, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_PHYSCHEM, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_JAZZY, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_ECFPCOUNTS, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_AVALON, _OC.GENERAL_PARAMETERS: {}},
                    {"name": _OC.DESCRIPTORS_MACCSKEYS, _OC.GENERAL_PARAMETERS: {}},
                    {
                        "name": _OC.DESCRIPTORS_PRECOMPUTED,
                        _OC.GENERAL_PARAMETERS: {
                            _OC.DESCRIPTORS_PRECOMPUTED_FILE: train_with_fp,
                            _OC.DESCRIPTORS_PRECOMPUTED_INPUT_COLUMNN: "canonical",
                            _OC.DESCRIPTORS_PRECOMPUTED_RESPONSE_COLUMN: "fp",
                        },
                    },
                ]
            },
        }
    ]

    # generate optimization configuration and execute it
    conf = deserialize(OptimizationConfig, confDict)
    train_smiles, train_y, _, _, _, _ = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)
