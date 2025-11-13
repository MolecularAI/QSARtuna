import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import optunaz.three_step_opt_build_merge
from optunaz import predict
from optunaz.config.buildconfig import (
    BuildConfig,
    ChemPropRegressor,
    MapieRegressor,
    RandomForestRegressor,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, SmilesFromFile
from optunaz.model_writer import ModelMode, Predictor, QptunaModel
from optunaz.utils.active_learning.similarity_to_train import descriptor_space_similarity

@pytest.fixture
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_test(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "test.csv")


class MockPredictorCls(Predictor):
    def __init__(self):
        self.train_smiles_ = np.array(["CCO", "CCN", "CCC"])
        self.X_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    def predict(self, data):
        return np.array([0.5] * len(data))

    def predict_proba(self, data):
        return np.array([[0.5, 0.5]] * len(data))

    def predict_uncert(self, data):
        return np.array([0.1] * len(data)), np.array([0.05] * len(data))

    def chemprop_fingerprint(self, data):
        return np.array([[0.1, 0.2, 0.3]] * len(data))


class MockPredictorReg(Predictor):
    def __init__(self):
        self.train_smiles_ = np.array(["CCO", "CCN", "CCC"])
        self.X_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    def predict(self, data):
        return np.array([0.1, 0.2])

    def predict_uncert(self, data):
        return np.array([0.1] * len(data)), np.array([0.05] * len(data))

    def chemprop_fingerprint(self, data):
        return np.array([[0.1, 0.2, 0.3]] * len(data))


def test_calc_sim_to_train_returns_correct_similarities():
    predictor = MockPredictorReg()
    descriptor = MagicMock()
    model = QptunaModel(predictor, descriptor, ModelMode.REGRESSION)
    descriptors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    nn_mol, nn_sim, nn_pred = descriptor_space_similarity(model, descriptors)
    assert np.all(nn_mol == ["CCO", "CCN"])
    assert np.allclose(nn_sim, [1.0, 1.0])
    assert np.allclose(nn_pred, [0.1, 0.2])


@pytest.fixture
def buildconfig_mapie(file_drd2_50, clean_shared_datadir):
    return BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
            test_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "test.csv"
            ),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=MapieRegressor.new(
            estimator=RandomForestRegressor.new(),
        ),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
        ),
    )


@pytest.fixture
def buildconfig_chemprop(file_drd2_50, clean_shared_datadir):
    return BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
        ),
        metadata=None,
        descriptor=SmilesFromFile.new(),
        algorithm=ChemPropRegressor.new(epochs=1, loss_function="mve"),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
        ),
    )


@pytest.mark.parametrize(
    "bcfg, sim_metric",
    [
        ("buildconfig_mapie", "ecfp_tanimoto"),
        ("buildconfig_mapie", "model_descriptor"),
        ("buildconfig_chemprop", "model_descriptor"),
    ],
)
def test_stt_identical(clean_shared_datadir, file_drd2_50, bcfg, sim_metric, request):
    bcfg = request.getfixturevalue(bcfg)

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_pkl:
        optunaz.three_step_opt_build_merge.build_best(bcfg, build_pkl.name)

    predict_args = [
        "prog",
        "--model-file",
        build_pkl.name,
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
        "--predict-uncertainty",
        "--sim_to_train",
        "--sim_metric",
        sim_metric
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(str(clean_shared_datadir / "outprediction"))

    assert np.all(predictions[f"{sim_metric.capitalize()}_nn_sim"] == 1)
    assert np.all(predictions["canonical"] == predictions["NN_molecule"])
    assert len(predictions.dropna()) == 50


@pytest.mark.parametrize(
    "bcfg, result, sim_metric",
    [
        ("buildconfig_mapie", 0.632421, "ecfp_tanimoto"),
        ("buildconfig_mapie", 0.377964, "model_descriptor"),
        ("buildconfig_chemprop", 0.999755, "model_descriptor"),
    ],
)
def test_stt_nonidentical(
    clean_shared_datadir, file_drd2_50, file_drd2_50_test, bcfg, result, request, sim_metric
):
    bcfg = request.getfixturevalue(bcfg)

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_pkl:
        optunaz.three_step_opt_build_merge.build_best(bcfg, build_pkl.name)

    predict_args = [
        "prog",
        "--model-file",
        build_pkl.name,
        "--input-smiles-csv-file",
        file_drd2_50_test,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
        "--predict-uncertainty",
        "--sim_to_train",
        "--sim_metric",
        sim_metric
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(str(clean_shared_datadir / "outprediction"))
    assert np.all(~(predictions[f"{sim_metric.capitalize()}_nn_sim"] == 1))

    assert np.isclose(predictions.loc[0, f"{sim_metric.capitalize()}_nn_sim"], result)
    assert len(predictions.dropna()) == 50
