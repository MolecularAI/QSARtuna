import pytest
import sklearn
from apischema import deserialize

from optunaz.builder import build
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils import load_json
from optunaz.utils.files_paths import attach_root_path


def getmodel(fname, respcol):
    confdict = load_json.loadJSON(path=attach_root_path(fname))

    data = confdict["data"]
    data["training_dataset_file"] = attach_root_path(
        "tests/data/DRD2/subset-50/train.csv"
    )
    if "test_dataset_file" in data and data["test_dataset_file"] is not None:
        data["test_dataset_file"] = attach_root_path(
            "tests/data/DRD2/subset-50/test.csv"
        )
    data["input_column"] = "canonical"
    data["response_column"] = respcol

    conf = deserialize(BuildConfig, confdict)
    model, train_score, test_score = build(conf)
    return model, train_score, test_score


def test_RF():
    model, train_score, test_score = getmodel(
        "examples/building/RF_regression_build.json", "molwt"
    )
    assert isinstance(model, sklearn.ensemble.forest.RandomForestRegressor)


def test_Ridge():
    model, train_score, test_score = getmodel(
        "examples/building/Ridge_regression_build.json", "molwt"
    )
    assert test_score is None
    assert isinstance(model, sklearn.linear_model.ridge.Ridge)


def test_SVC():
    model, train_score, test_score = getmodel(
        "examples/building/classification_build.json", "molwt_gt_330"
    )
    assert isinstance(model, sklearn.svm.SVC)
