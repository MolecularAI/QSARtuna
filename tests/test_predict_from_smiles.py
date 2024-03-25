import pickle
import tempfile
from dataclasses import dataclass

import numpy as np
import numpy.testing as npt

from optunaz.config import ModelMode
from optunaz.descriptors import ECFP, SmilesFromFile
from optunaz.model_writer import (
    save_model,
    QSARtunaModel,
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
        return np.array([len(x[0]) for x in xs])


@dataclass
class BuildConfig:
    pass


@dataclass
class Settings:
    pass


def test1():
    descriptor = ECFP.new()
    m0 = Model()
    mode = ModelMode.REGRESSION
    m = QSARtunaModel(m0, descriptor, mode)

    smis = ["CCC"]
    assert m.predict_from_smiles(smis) == np.array([4])

    smis_bad = ["CCC", "xyz"]
    npt.assert_equal(m.predict_from_smiles(smis_bad), [4, np.nan])


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
        loaded = pickle.load(open(f.name, "rb"))

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
        loaded = pickle.load(open(f.name, "rb"))

    smis = ["CCC"]
    assert loaded.predict_from_smiles(smis) == np.array([3])
