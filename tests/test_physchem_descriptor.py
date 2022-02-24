import tempfile

import jsonpickle
import numpy.testing as npt
import pytest
import sklearn

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    RandomForest,
    PLS,
    XGBregressor,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    PhyschemDescriptors,
    UnfittedSklearnScaler,
    FittedSklearnScaler, ScaledDescriptor,
)


def test_physchem_descriptor(shared_datadir):
    file_drd2_300 = str(shared_datadir / "DRD2" / "subset-300" / "train.csv")
    config = OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=file_drd2_300,
        ),
        descriptors=[
            PhyschemDescriptors.new()
        ],
        algorithms=[RandomForest.new(), XGBregressor.new(), PLS.new()],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=15,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )
    optunaz.three_step_opt_build_merge.optimize(config, "test_regression_physchem")


def test_physchem_scaler(shared_datadir):
    p = shared_datadir / "DRD2/subset-50/train.csv"
    d = ScaledDescriptor.new(
        descriptor=PhyschemDescriptors.new(),
        scaler=UnfittedSklearnScaler(
            mol_data=UnfittedSklearnScaler.MolData(
                file_path=str(p), smiles_column="canonical"
            )
        )
    )

    # Get a SMILES string on which Scaler was fitted.
    # "CCC" is way off from the training space.
    import pandas as pd

    df = pd.read_csv(p)
    smi = df["canonical"][0]

    pred = d.calculate_from_smi(smi)
    npt.assert_almost_equal(pred[0:3], [0.16, 0.60, 0.16,], 2)


def test_physchem_noscaler(shared_datadir):
    d = PhyschemDescriptors.new()
    pred = d.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [2.125, 1.25, 2.125], 2)


@pytest.fixture
def mol_set1():
    return ["CCC", "CCCCC", "c1ccccc1"]


def test_set1_noscaler(mol_set1):
    d = PhyschemDescriptors.new()
    pred = d.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [2.12, 1.25, 2.12], 2)


def test_set1_scaler_simple(mol_set1):
    with tempfile.NamedTemporaryFile(mode="w+t") as f:
        import pandas as pd

        df = pd.DataFrame({"SMILES": mol_set1})
        df.to_csv(f.name)
        d = ScaledDescriptor.new(
            descriptor=PhyschemDescriptors.new(),
            scaler=UnfittedSklearnScaler(
                mol_data=UnfittedSklearnScaler.MolData(
                    file_path=f.name, smiles_column="SMILES"
                )
            )
        )
    pred = d.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [0.15, -0.84, 0.15], 2)


def test_set1_scaler_advanced(mol_set1):
    d = PhyschemDescriptors.new()
    x = [d.calculate_from_smi(m) for m in mol_set1]
    scaler = sklearn.preprocessing.MaxAbsScaler()
    scaler.fit(x)
    d2 = ScaledDescriptor.new(
        descriptor=PhyschemDescriptors.new(),
        scaler=FittedSklearnScaler(saved_params=jsonpickle.dumps(scaler))
    )
    pred = d2.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [0.96, 0.62, 0.96], 2)


def test_serializeation(shared_datadir):
    p = shared_datadir / "DRD2/subset-50/train.csv"
    d = ScaledDescriptor.new(
        descriptor=PhyschemDescriptors.new(),
        scaler=UnfittedSklearnScaler(
            mol_data=UnfittedSklearnScaler.MolData(
                file_path=str(p), smiles_column="canonical"
            )
        )
    )
    from apischema import serialize
    serialize(d)
