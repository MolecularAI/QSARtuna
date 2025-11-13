import numpy.testing as npt

from optunaz.algorithms.chem_prop import ChemPropRegressor
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    SmilesAndSideInfoFromFile,
    SmilesFromFile,
    combine_covariates,
)
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.utils.preprocessing.transform import ZScales


def test_chemprop_trainpred(clean_shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=clean_shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    X, y, _, _, _, _ = dataset.get_sets()
    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1, message_hidden_dim=message_hidden_dim, ffn_hidden_dim=ffn_hidden_dim
    )
    reg.fit(X, y)
    preds = reg.predict(
        [
            ["C"],
            ["NC(=O)C1=C2CCC3=C(SN=C3)C2=C(OC2=CC=CC=C2)S1"],
            ["O=O"],
        ]
    )
    npt.assert_allclose(
        preds, [0.47488937, 0.4989107, 0.47533718], rtol=1e-05, atol=1e-05
    )
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    intrprt = reg.interpret([["C"], ["CC"], ["CCC"]], prop_delta=0.1, min_atoms=1)

    assert intrprt.dropna().shape == (1, 4)
    assert intrprt.iloc[2]["rationale_0"] == "C[CH3:1]"
    assert mpn_fps.shape == (len(X), message_hidden_dim)
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)


def test_chemprop_morgan_xaux(clean_shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=clean_shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
        covariate_column="PredefinedSplit",
    )
    X, y, x_aux, _, _, _ = dataset.get_sets()
    smi = SmilesAndSideInfoFromFile.new(
        file=str(clean_shared_datadir) + "/pxc50/P24863.csv",
        input_column="Smiles",
        x_aux_column="PredefinedSplit",
    )
    descriptors = smi.parallel_compute_descriptor(X)
    X = combine_covariates(smi, descriptors, x_aux)

    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1,
        message_hidden_dim=message_hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        molecule_featurizers=["morgan_binary"],
    )
    reg.fit(X, y)
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    assert mpn_fps.shape == (len(X), message_hidden_dim + 2048 + 2)  # +2 for x_aux & covariate covariate_column
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)


def test_chemprop_maccs_yaux(clean_shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=clean_shared_datadir / "pxc50/P24863.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
        covariate_column="PredefinedSplit",
    )
    X, y, x_aux, _, _, _ = dataset.get_sets()
    smi = SmilesAndSideInfoFromFile.new(
        file=str(clean_shared_datadir) + "/pxc50/P24863.csv",
        input_column="Smiles",
        y_aux_column="PredefinedSplit",
    )
    descriptors = smi.parallel_compute_descriptor(X)
    X = combine_covariates(smi, descriptors, x_aux)

    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1,
        message_hidden_dim=message_hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        molecule_featurizers=["maccs_keys"],
    )
    reg.fit(X, y)
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    assert mpn_fps.shape == (len(X), message_hidden_dim + 168) # 167 for maccs + 1 covariate_column
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)


def test_chemprop_avalon_zscales(clean_shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Class",
        probabilistic_threshold_representation=True,
        probabilistic_threshold_representation_threshold=8,
        probabilistic_threshold_representation_std=0.6,
        training_dataset_file=clean_shared_datadir
        / "peptide/toxinpred3/subset-50/train.csv",
        deduplication_strategy=KeepAllNoDeduplication(),
        covariate_column="Peptide",
        aux_transform=ZScales.new(),
    )
    X, y, x_aux, _, _, _ = dataset.get_sets()
    smi = SmilesAndSideInfoFromFile.new(
        file=str(clean_shared_datadir) + "/peptide/toxinpred3/subset-50/train.csv",
        input_column="Smiles",
        x_aux_column="Class",
    )
    descriptors = smi.parallel_compute_descriptor(X)
    X = combine_covariates(smi, descriptors, x_aux)

    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1,
        message_hidden_dim=message_hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        molecule_featurizers=["avalon"],
    )
    reg.fit(X, y)
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    assert mpn_fps.shape == (len(X), message_hidden_dim + 2048 + 6) # 5 zscales + 1 x_aux_column
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)

    # make the side info y labels now
    X, y, x_aux, _, _, _ = dataset.get_sets()
    smi = SmilesAndSideInfoFromFile.new(
        file=str(clean_shared_datadir) + "/peptide/toxinpred3/subset-50/train.csv",
        input_column="Smiles",
        y_aux_column="Class",
    )
    descriptors = smi.parallel_compute_descriptor(X)
    X = combine_covariates(smi, descriptors, x_aux)

    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1,
        message_hidden_dim=message_hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        molecule_featurizers=["maccs_keys"],
    )
    reg.fit(X, y)
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    assert mpn_fps.shape == (len(X), message_hidden_dim + 167 + 5) # 5 zscales
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)


def test_mordred(clean_shared_datadir):
    dataset = Dataset(
        input_column="Smiles",
        response_column="Measurement",
        training_dataset_file=clean_shared_datadir / "pxc50/P24863.csv",
    )
    X, y, x_aux, _, _, _ = dataset.get_sets()
    smi = SmilesFromFile.new()
    descriptors = smi.parallel_compute_descriptor(X)

    message_hidden_dim = 350
    ffn_hidden_dim = 500
    reg = ChemPropRegressor(
        epochs=1,
        message_hidden_dim=message_hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        molecule_featurizers=["mordreddescriptors"],
    )
    reg.fit(X, y)
    mpn_fps = reg.chemprop_fingerprint(X, fingerprint_type="MPN")
    last_FFN_fps = reg.chemprop_fingerprint(X, fingerprint_type="last_FFN")
    assert mpn_fps.shape == (len(X), message_hidden_dim + 3226)
    assert last_FFN_fps.shape == (len(X), ffn_hidden_dim)