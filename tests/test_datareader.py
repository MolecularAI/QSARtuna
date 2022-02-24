import pytest

from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepFirst
from optunaz.utils.preprocessing.splitter import Random, Stratified


@pytest.fixture
def drd2_300(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-300" / "train.csv")


def test_nosplit(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
    )
    train_smiles, train_y, test_smiles, test_y = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 300
    assert len(test_smiles) == 0


def test_split(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, test_smiles, test_y = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60


def test_split_and_extra(drd2_300, shared_datadir):
    testfile = str(shared_datadir / "DRD2" / "subset-50" / "test.csv")
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        test_dataset_file=testfile,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, test_smiles, test_y = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60 + 50


def test_split_strat(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Stratified(fraction=0.2, seed=42),
    )
    train_smiles, train_y, test_smiles, test_y = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60


