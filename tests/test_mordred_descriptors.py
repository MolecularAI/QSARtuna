import json
import sys
from unittest.mock import patch
import pytest
from apischema import deserialize
from optunaz import optbuild, predict
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils.files_paths import attach_root_path


@pytest.fixture
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def cls_config(clean_shared_datadir):
    """Returns the opt config."""
    return f"{attach_root_path('examples/optimization/classification.json')}"


@pytest.fixture
def cls_config2(clean_shared_datadir):
    """Returns a build config."""
    return f"{attach_root_path('examples/building/classification_build.json')}"

def test_mordred_opt(clean_shared_datadir, file_drd2_50, cls_config):
    testargs = [
        "prog",
        "--config",
        str(cls_config),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        str(file_drd2_50),
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()



def test_mordred_build(clean_shared_datadir, file_drd2_50, cls_config2):
    testargs = [
        "prog",
        "--config",
        str(cls_config2),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(clean_shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        str(file_drd2_50),
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(clean_shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()