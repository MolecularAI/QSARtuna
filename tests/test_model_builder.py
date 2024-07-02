import pytest
import sklearn

from apischema import deserialize

from optunaz.algorithms import chem_prop_hyperopt, calibrated_cv
from optunaz.builder import build
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils import load_json
from optunaz.utils.files_paths import attach_root_path
import pickle


def getmodel(fname, respcol):
    confdict = load_json.loadJSON(path=attach_root_path(fname))

    data = confdict["data"]
    data["training_dataset_file"] = attach_root_path(data["training_dataset_file"])
    if "test_dataset_file" in data and data["test_dataset_file"] is not None:
        data["test_dataset_file"] = attach_root_path(data["test_dataset_file"])
    conf = deserialize(BuildConfig, confdict)

    model, train_score, test_score = build(conf)
    return model, train_score, test_score


def test_ThrowClsResponseColErrors():
    try:
        getmodel(
            "examples/building/classification_build_handle_response_issue.json",
            "molwt_gt_330",
        )
    except Exception as e:
        assert isinstance(e, ValueError)


def test_ThrowRegResponseColErrors():
    try:
        getmodel(
            "examples/building/RF_regression_build_handle_response_issue.json",
            "molwt",
        )
    except Exception as e:
        assert isinstance(e, ValueError)


def test_ThrowXGBSmileErrors():
    try:
        getmodel(
            "examples/building/invalid_xgb_smilesfromfile.json",
            "molwt",
        )
    except Exception as e:
        assert isinstance(e, ValueError)


def test_ThrowChemPropECFPErrors():
    try:
        getmodel(
            "examples/building/invalid_chemprop_ecfp.json",
            "molwt",
        )
    except Exception as e:
        assert isinstance(e, ValueError)


def test_RF():
    model, train_score, test_score = getmodel(
        "examples/building/RF_regression_build.json", "molwt"
    )
    assert isinstance(model, sklearn.ensemble.RandomForestRegressor)


def test_Ridge():
    model, train_score, test_score = getmodel(
        "examples/building/Ridge_regression_build.json", "molwt"
    )
    assert test_score is None
    assert isinstance(model, sklearn.linear_model.Ridge)


def test_SVC():
    model, train_score, test_score = getmodel(
        "examples/building/classification_build.json", "molwt_gt_330"
    )
    assert isinstance(model, sklearn.svm.SVC)


def test_ChemPropHyperoptRegressor():
    model, train_score, test_score = getmodel(
        "examples/building/ChemProp_regression_build.json", "molwt"
    )
    assert isinstance(model, chem_prop_hyperopt.ChemPropHyperoptRegressor)


def test_Calibration():
    model, train_score, test_score = getmodel(
        "examples/building/calibration_build.json", "molwt_gt_330"
    )
    assert isinstance(model, calibrated_cv.CalibratedClassifierCVWithVA)


def test_Calibration_ChemProp(shared_datadir):
    model, train_score, test_score = getmodel(
        "examples/building/calibration_chemprop_build.json", "molwt_gt_330"
    )
    assert isinstance(model, calibrated_cv.CalibratedClassifierCVWithVA)
    with open(str(shared_datadir / "test.pkl"), "wb") as fid:
        pickle.dump(model, fid)