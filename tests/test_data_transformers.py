import numpy as np
import pytest
import scipy
from scipy.stats import norm
from optunaz.utils.preprocessing.transform import (
    LogBase,
    LogNegative,
    ModelDataTransform,
    PTRTransform,
    ZScales
)
from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepFirst
from optunaz.utils.preprocessing.splitter import Random
import numpy.testing as npt


@pytest.fixture
def drd2_50(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")

@pytest.fixture
def peptide_toxinpred3(shared_datadir):
    return str(shared_datadir / "peptide" / "toxinpred3" / "train.csv")


@pytest.mark.parametrize(
    "logbase",
    [LogBase.LOG2, LogBase.LOG10, LogBase.LOG],
)
@pytest.mark.parametrize(
    "lognegative",
    [
        LogNegative.TRUE,
        LogNegative.FALSE,
    ],
)
@pytest.mark.parametrize(
    "log_transform_unit_conversion",
    [
        None,
        5,
        6,
    ],
)
def test_log_transform(drd2_50, logbase, lognegative, log_transform_unit_conversion):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_50,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    transform = ModelDataTransform.new(
        base=logbase,
        negation=lognegative,
        conversion=log_transform_unit_conversion,
    )
    transformed = transform.transform(train_y)
    reverse_transform = transform.reverse_transform(transformed)
    npt.assert_allclose(train_y, reverse_transform, rtol=1e-05, atol=1e-05)


def test_pxc50_values():
    data = np.array([1000, 100, 10, 1, 0.1])
    transform = ModelDataTransform.new(
        base=LogBase.LOG10,
        negation=LogNegative.TRUE,
        conversion=6,
    )
    data_t = transform.transform(data)
    assert all(data_t == np.array([3.0, 4.0, 5.0, 6.0, 7.0]))


def test_ptr_transform(drd2_50):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_50,
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    ptr_transformed = PTRTransform.new(threshold=300, std=20)
    transformed = ptr_transformed.transform(train_y)
    min_val = norm.ppf(0.0000000000000001, 300, 20)
    max_val = norm.ppf(0.9999999999999999, 300, 20)
    reverse_transform = ptr_transformed.reverse_transform(transformed)
    train_y = train_y.clip(min_val, max_val)
    npt.assert_allclose(train_y, reverse_transform, rtol=1e-04, atol=1e-06)


def test_zscales(peptide_toxinpred3):
    data = Dataset(
        input_column="Smiles",
        response_column="Class",
        training_dataset_file=peptide_toxinpred3,
        aux_column="Peptide",
        aux_transform=ZScales.new()
    )
    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = data.get_sets()
    assert train_aux.shape == (8828, 5)
