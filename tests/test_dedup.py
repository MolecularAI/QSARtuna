import pandas as pd
import pytest
from optunaz.utils.preprocessing.deduplicator import *


@pytest.fixture()
def dupdf():
    df = pd.DataFrame(
        {
            "SMILES": ["CCC", "CCC", "CCC", "CCC", "CCCCC", "CCCCC"],
            "val": [3, 10, 1, 7, 4, 5]
        }
    )
    return df


@pytest.fixture()
def dupdf_cat():
    df = pd.DataFrame(
        {
            "SMILES": ["CCC", "CCC", "CCC", "CCC", "CCCCC", "CCCCC"],
            "val": ["A", "A", "C", "B", "C", "C"]
        }
    )
    return df


def test_first(dupdf):
    df = KeepFirst.dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [3, 4]


def test_last(dupdf):
    df = KeepLast.dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [7, 5]


def test_random(dupdf):
    df = KeepRandom(seed=42).dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [3, 5]

    df = KeepRandom(seed=73).dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [4, 10]


def test_min(dupdf):
    df = KeepMin.dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [1, 4]


def test_max(dupdf):
    df = KeepMax.dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [10, 5]


def test_avg_mean(dupdf):
    df = KeepAvg.dedup(dupdf, "SMILES")
    assert df["val"].tolist() == [5.25, 4.5]
