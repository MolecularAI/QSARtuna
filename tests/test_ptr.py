from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
import numpy as np


def test1(shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=7,
        probabilistic_threshold_representation_std=0.2,
        training_dataset_file=shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    _, y, _, _, _, _ = dataset.get_sets()
    assert np.allclose(y[0], 1.0)
    assert np.allclose(y[-1], 0.0)
    assert np.allclose(y[231], 0.5)
