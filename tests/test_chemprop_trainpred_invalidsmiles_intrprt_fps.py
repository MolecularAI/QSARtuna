from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.algorithms.chem_prop import ChemPropRegressor
import numpy as np


def test_chemprop_trainpred_invalidsmiles(shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    X, y, _, _, _, _ = dataset.get_sets()
    hidden_size = 350
    ffn_hidden_size = 500
    reg = ChemPropRegressor(
        epochs=4, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size
    )
    reg.fit(X, y)
    preds = reg.predict(
        [
            ["C"],
            ["NC(=O)C1=C2CCC3=C(SN=C3)C2=C(OC2=CC=CC=C2)S1"],
            ["nan"],
            ["error"],
            ["Invalid SMILES"],
            ["O=O"],
        ]
    )
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    intrprt = reg.interpret([["C"], ["CC"], ["CCC"]])
    assert all(np.isnan(preds[[2, 3, 4]].astype(float).flatten()))
    assert all(np.invert(np.isnan(preds[[0, 1, 5]].astype(float).flatten())))
    assert len(intrprt) == 3
    assert mpn_fps.shape == (len(X), hidden_size)
    assert last_FFN_fps.shape == (len(X), ffn_hidden_size)


def test_chemprop_trainpred_invalidsmiles_auxcol(shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
        aux_column="PredefinedSplit",
    )
    X, y, x_aux, _, _, _ = dataset.get_sets()
    X = [[i, [None]] for i in X]
    X = np.hstack((X, np.array(x_aux).reshape(len(x_aux), 1)))

    hidden_size = 350
    ffn_hidden_size = 500
    reg = ChemPropRegressor(
        epochs=4, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size
    )
    reg.fit(X, y)
    preds = reg.predict(
        [
            ["C"],
            ["NC(=O)C1=C2CCC3=C(SN=C3)C2=C(OC2=CC=CC=C2)S1"],
            ["nan"],
            ["error"],
            ["Invalid SMILES"],
            ["O=O"],
        ]
    )
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    intrprt = reg.interpret([["C"], ["CC"], ["CCC"]])
    assert all(np.isnan(preds[[2, 3, 4]].astype(float).flatten()))
    assert all(np.invert(np.isnan(preds[[0, 1, 5]].astype(float).flatten())))
    assert len(intrprt) == 3
    assert mpn_fps.shape == (len(X), hidden_size)
    assert last_FFN_fps.shape == (len(X), ffn_hidden_size)
