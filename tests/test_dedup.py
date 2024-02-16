import json

import apischema
import pandas as pd
import pytest
import numpy.testing as npt

from optunaz.config.optconfig import OptimizationConfig
from optunaz.datareader import Dataset, deduplicate
from optunaz.utils.preprocessing.deduplicator import *


@pytest.fixture()
def dupdf():
    df = pd.DataFrame(
        {
            "SMILES": ["CCC", "CCC", "CCC", "CCC", "CCCCC", "CCCCC"],
            "val": [3, 10, 1, 7, 4, 5],
        }
    )
    return df


@pytest.fixture()
def dupdf_cat():
    df = pd.DataFrame(
        {
            "SMILES": ["CCC", "CCC", "CCC", "CCC", "CCCCC", "CCCCC"],
            "val": [True, True, False, False, True, True],
        }
    )
    return df


def test_first(dupdf):
    df = KeepFirst().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [3, 4]


def test_last(dupdf):
    df = KeepLast().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [7, 5]


def test_random(dupdf):
    df = KeepRandom(seed=1).dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [7, 4]

    df = KeepRandom(seed=42).dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [10, 5]


def test_min(dupdf):
    df = KeepMin().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [1, 4]


def test_max(dupdf):
    df = KeepMax().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [10, 5]


def test_avg_mean(dupdf):
    df = KeepAvg().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [5.25, 4.5]


def test_median_reg(dupdf):
    df = KeepMedian().dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [5, 4.5]


def test_median_cls(dupdf_cat):
    smiles, y, _, groups = deduplicate(
        dupdf_cat["SMILES"],
        dupdf_cat["val"],
        None,
        None,
        KeepMedian(),
        "classification",
    )
    assert y.tolist() == [True, True]


def test_avg_mean_datareader_cls(dupdf_cat):
    smiles, y, _, groups = deduplicate(
        dupdf_cat["SMILES"], dupdf_cat["val"], None, None, KeepAvg(), "classification"
    )
    assert y.tolist() == [True, True]


def test_avg_mean_datareader_reg(dupdf):
    smiles, y, _, groups = deduplicate(
        dupdf["SMILES"], dupdf["val"], None, None, KeepAvg(), "regression"
    )
    assert y.tolist() == [5.25, 4.5]


def test_avg_mean_datareader_reg_aux(dupdf):
    smiles, y, _, groups = deduplicate(
        dupdf["SMILES"], dupdf["val"], [10] * 3 + [5] * 3, None, KeepAvg(), "regression"
    )
    npt.assert_allclose(y.tolist(), [4.5, 4.6666, 7], rtol=1e-04, atol=1e-04)


def test_config(shared_datadir):
    confstr = """
    {
    "data": {
        "deduplication_strategy": {
            "name": "KeepAvg"
        },
        "split_strategy": {
            "name": "Random",
            "fraction": 0.2,
            "seed": 1
        },
        "save_intermediate_files": true,
        "training_dataset_file": "train.csv",
        "input_column": "canonical",
        "response_column": "molwt"
    },
    "algorithms": [
        {
            "parameters": {
                "C": {
                    "low": 1e-10,
                    "high": 100
                },
                "gamma": {
                    "low": 0.0001,
                    "high": 100
                }
            },
            "name": "SVR"
        },
        {
            "parameters": {
                "max_depth": {
                    "low": 2,
                    "high": 32
                },
                "n_estimators": {
                    "low": 10,
                    "high": 250
                },
                "max_features": [
                    "auto"
                ]
            },
            "name": "RandomForestRegressor"
        }
    ],
    "descriptors": [
        {
            "parameters": {
                "nBits": 2048,
                "radius": 3,
                "useFeatures": true
            },
            "name": "ECFP_counts"
        }
    ],
    "settings": {
        "cross_validation": 5,
        "n_trials": 150,
        "n_jobs": -1,
        "n_startup_trials": 50,
        "track_to_mlflow": false,
        "random_seed": 785
    },
    "task": "optimization"
}
    """
    config = apischema.deserialize(OptimizationConfig, json.loads(confstr))
    config.data.training_dataset_file = str(
        shared_datadir / "DRD2" / "subset-50" / "train.csv"
    )
    config.data.get_sets()
