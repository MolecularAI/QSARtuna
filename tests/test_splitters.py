import pytest
import pandas as pd
from optunaz.utils.preprocessing.splitter import *


@pytest.fixture()
def dataset():
    df = pd.DataFrame(
        {
            "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "val": [3, 10, 1, 7, 4],
        }
    )
    return df


def test_random(dataset):
    train, test = Random(fraction=0.2, seed=42).split(dataset)
    assert len(train) == 4
    assert len(test) == 1
    assert test.iloc[0, 1] == 10

    train, test = Random(fraction=0.4, seed=74).split(dataset)
    assert len(train) == 3
    assert len(test) == 2
    assert test.iloc[0, 1] == 1


def test_temporal(dataset):
    train, test = Temporal(fraction=0.4).split(dataset)
    assert len(train) == 3
    assert len(test) == 2
    assert test["val"].tolist() == [7, 4]


def test_stratified(dataset):
    train, test = Stratified(fraction=0.4, seed=42, respcol="val").split(dataset)
    assert len(train) == 3
    assert len(test) == 2
    assert test["val"].tolist() == [7, 3]
