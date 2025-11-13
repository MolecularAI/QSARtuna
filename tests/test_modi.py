import numpy as np
import pytest

from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.utils.preprocessing.modi import (
    assess_modelability,
    calculate_classification_modi_metrics,
    calculate_regression_modi_metrics,
)
from optunaz.utils.preprocessing.splitter import Random


@pytest.fixture
def drd2_300(clean_shared_datadir):
    return str(clean_shared_datadir / "DRD2" / "subset-300" / "train.csv")


@pytest.fixture
def drd2_reg_dataset(drd2_300, clean_shared_datadir):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepAllNoDeduplication(),
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = data.get_sets()
    return train_smiles, train_y, train_aux, test_smiles, test_y, test_aux


@pytest.fixture
def drd2_cls_dataset(drd2_300, clean_shared_datadir):
    data = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepAllNoDeduplication(),
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = data.get_sets()
    return train_smiles, train_y, train_aux, test_smiles, test_y, test_aux


def test_calculate_classification_modi_metrics_with_valid_input():
    labels = np.array([0, 1, 0, 0, 1, 0, 1])
    descriptors = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
    )
    result = calculate_classification_modi_metrics(labels, descriptors)
    assert len(result) == 4
    assert all(isinstance(x, float) for x in result)


def test_calculate_regression_modi_metrics_with_valid_input():
    labels = np.array([0.2, 1.2, 2.2, 3.3, 4.2, 5.2, 6.2])
    descriptors = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
    result = calculate_regression_modi_metrics(labels, descriptors)
    assert len(result) == 5
    assert all(isinstance(x, float) for x in result)


def test_assess_modelability_with_classification(drd2_cls_dataset):
    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = drd2_cls_dataset
    result = assess_modelability(
        train_smiles,
        train_y,
        train_aux,
        test_smiles,
        test_y,
        test_aux,
        "classification",
    )
    assert "modi_div" in result
    assert "modi_aci" in result
    assert "modi_ccr" in result


def test_assess_modelability_with_regression(drd2_reg_dataset):
    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = drd2_reg_dataset
    result = assess_modelability(
        train_smiles, train_y, train_aux, test_smiles, test_y, test_aux, "regression"
    )
    assert "modi_div" in result
    assert "modi_q2" in result
    assert "modi_ssR2" in result


def test_assess_modelability_with_invalid_response_type():
    train_smiles = ["CCO", "CCC"]
    train_y = np.array([1.0, 2.0])
    train_aux = np.array([[1, 2], [3, 4]])
    test_smiles = ["CCN"]
    test_y = np.array([3.0])
    test_aux = np.array([[5, 6]])
    result = assess_modelability(
        train_smiles, train_y, train_aux, test_smiles, test_y, test_aux, "invalid"
    )
    assert result is None
