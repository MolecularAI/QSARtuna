import dill
import tempfile
from dataclasses import dataclass

import numpy as np
import numpy.testing as npt

from optunaz.config import ModelMode
from optunaz.descriptors import ECFP, SmilesFromFile, UnscaledJazzyDescriptors
from optunaz.model_writer import (
    save_model,
    QptunaModel,
    Predictor,
)


@dataclass
class Model(Predictor):
    def predict(self, xs):
        return np.sum(xs)


@dataclass
class ChempropModel(Predictor):
    """This model returns length of SMILES string as a prediction."""

    def predict(self, xs):
        return np.array([len(x) for x in xs])


@dataclass
class BuildConfig:
    pass


@dataclass
class Settings:
    pass


def test_ecfp():
    descriptor = ECFP.new()
    m0 = Model()
    m0.predict_uncert = lambda x: np.array([1])
    m0.X_ = np.array([descriptor.calculate_from_smi("C")])
    m0.train_smiles_ = np.array(["C"])
    mode = ModelMode.REGRESSION
    m = QptunaModel(m0, descriptor, mode)

    smis = ["CCC"]
    test_result = m.predict_from_smiles(smis, sim_to_train=True, uncert=True)
    assert test_result[0] == np.array([4])
    assert test_result[1] == np.array([1])
    assert test_result[3] == np.array(["C"])

    smis_bad = ["CCC", "xyz"]
    npt.assert_equal(m.predict_from_smiles(smis_bad), [4, np.nan])


def test_jazzy():
    descriptor = UnscaledJazzyDescriptors.new()
    m0 = Model()
    m0.predict_uncert = lambda x: np.array([1])
    m0.X_ = np.array([descriptor.calculate_from_smi("C")])
    m0.train_smiles_ = np.array(["C"])
    mode = ModelMode.REGRESSION
    m = QptunaModel(m0, descriptor, mode)

    smis = ["CCC"]
    test_result = m.predict_from_smiles(smis, sim_to_train=True, uncert=True)
    assert test_result[1] == np.array([1])
    assert test_result[3] == np.array(["C"])

    smis_bad = ["CCC", "xyz"]
    npt.assert_equal(m.predict_from_smiles(smis_bad)[1], np.nan)


def test_unpickling_sklearn():
    model = Model()
    buildconfig = BuildConfig()
    buildconfig.settings = Settings()
    buildconfig.settings.mode = ModelMode.REGRESSION
    buildconfig.descriptor = ECFP.new()
    train_scores = float("nan")
    test_scores = float("nan")

    with tempfile.NamedTemporaryFile("wb") as f:
        save_model(model, buildconfig, f.name, train_scores, test_scores)
        loaded = dill.load(open(f.name, "rb"))

    smis = ["CCC"]
    assert loaded.predict_from_smiles(smis) == np.array([4])


def test_unpickling_chemprop():
    model = ChempropModel()
    buildconfig = BuildConfig()
    buildconfig.settings = Settings()
    buildconfig.settings.mode = ModelMode.REGRESSION
    buildconfig.descriptor = SmilesFromFile.new()
    train_scores = float("nan")
    test_scores = float("nan")

    with tempfile.NamedTemporaryFile("wb") as f:
        save_model(model, buildconfig, f.name, train_scores, test_scores)
        loaded = dill.load(open(f.name, "rb"))

    smis = ["CCC"]
    assert loaded.predict_from_smiles(smis) == np.array([3])
