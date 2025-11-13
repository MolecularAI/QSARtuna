import numpy as np
import pytest

from optunaz.algorithms.fast_prop import FastPropClassifier, FastPropRegressor
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    UnscaledPhyschemDescriptors,
    descriptor_from_config,
)


@pytest.fixture
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


def test_fastprop_trainpred_intrprt(file_drd2_50):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=file_drd2_50,
    )
    X, y, _, _, _, _ = dataset.get_sets()
    descriptor, _ = descriptor_from_config(X, UnscaledPhyschemDescriptors.new())

    reg = FastPropRegressor(number_epochs=1, patience=0)
    reg.fit(descriptor, y)
    preds, unc = reg.predict_uncert(descriptor)

    assert preds.shape == (50,)
    assert not np.all(unc)


def test_fastprop_trainpred_cls(file_drd2_50):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        training_dataset_file=file_drd2_50,
    )
    X, y, _, _, _, _ = dataset.get_sets()
    descriptor, _ = descriptor_from_config(X, ECFP.new())

    cls = FastPropClassifier(number_epochs=1, patience=0)
    cls.fit(descriptor, y)
    preds, unc = cls.predict_uncert(descriptor)
    assert preds.shape == (50,)
    assert not np.all(unc)
