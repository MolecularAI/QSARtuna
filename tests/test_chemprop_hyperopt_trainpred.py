import pytest

from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.algorithms.chem_prop_hyperopt import (
    ChemPropHyperoptClassifier,
    ChemPropHyperoptRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy.testing as npt


@pytest.mark.parametrize(
    "features_generator,expected",
    [
        ("none", [0.746169, 0.763537]),
        ("rdkit_2d_normalized", [0.913036, 0.84646]),
    ],
)
def test_chemprop_classifier_trainpred(shared_datadir, features_generator, expected):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        training_dataset_file=shared_datadir / "DRD2/subset-50/train.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    X, y, _, _, _, _ = dataset.get_sets()
    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    cls = ChemPropHyperoptClassifier(
        epochs=4,
        num_iters=2,
        search_parameter_level="auto",
        features_generator=features_generator,
    )
    scores = cross_val_score(cls, X, y, cv=kf, scoring="average_precision")
    npt.assert_allclose(scores, expected, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize(
    "features_generator,expected",
    [
        ("none", [-0.06824469, 0.04185276]),
        ("rdkit_2d_normalized", [-0.28931212, 0.09185344]),
    ],
)
def test_chemprop_regressor_trainpred(shared_datadir, features_generator, expected):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=shared_datadir / "DRD2/subset-50/train.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    X, y, _, _, _, _ = dataset.get_sets()
    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    reg = ChemPropHyperoptRegressor(
        epochs=4,
        num_iters=2,
        search_parameter_level="auto",
        features_generator=features_generator,
    )
    scores = cross_val_score(reg, X, y, cv=kf, scoring="r2")
    npt.assert_allclose(scores, expected, rtol=1e-04, atol=1e-04)
