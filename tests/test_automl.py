import os
import sys
from unittest.mock import patch
import pytest
from optunaz import automl
from optunaz.utils.files_paths import attach_root_path
import shutil
import pandas as pd
import json


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def drd2_pkl(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "drd2_reg.pkl")


@pytest.fixture
def slurm_template(shared_datadir):
    """Returns the example slurm template."""
    return f"{attach_root_path('examples/slurm-scripts/automl.template')}"


@pytest.fixture
def initial_config(shared_datadir):
    """Returns the config for a first run."""
    return f"{attach_root_path('examples/automl/config.initial.template')}"


@pytest.fixture
def retrain_config(shared_datadir):
    """Returns the config for a subsequent run."""
    return f"{attach_root_path('examples/automl/config.retrain.template')}"


@pytest.mark.parametrize(
    "taskids",
    ["one_taskid", "two_taskid"],
)
def test_init_automl(
    taskids,
    file_drd2_50,
    drd2_pkl,
    slurm_template,
    initial_config,
    retrain_config,
    shared_datadir,
):
    automl_args = [
        "prog",
        "--output-path",
        str(shared_datadir / "output_path"),
        "--email",
        "test@test.com",
        "--user_name",
        "test",
        "--input-data",
        str(shared_datadir / "automl/*"),
        "--input-smiles-csv-column",
        "canonical",
        "--input-activity-csv-column",
        "molwt",
        "--input-task-csv-column",
        taskids,
        "--input-initial-template",
        str(initial_config),
        "--input-retrain-template",
        str(retrain_config),
        "--input-slurm-template",
        str(slurm_template),
        "--n-cores",
        "1",
        "--dry-run",
        "-vvv",
        "--slurm-al-pool",
        str(file_drd2_50),
        "--slurm-al-smiles-csv-column",
        "canonical",
        "--slurm-job-prefix",
        "testaml"
    ]
    with patch.object(sys, "argv", automl_args):
        automl.main()


def test_automl_fullflow(
    file_drd2_50,
    drd2_pkl,
    slurm_template,
    initial_config,
    retrain_config,
    shared_datadir,
):
    automl_args = [
        "prog",
        "--output-path",
        str(shared_datadir / "output_path"),
        "--email",
        "test@test.com",
        "--user_name",
        "test",
        "--input-data",
        str(shared_datadir / "automl/*"),
        "--input-smiles-csv-column",
        "canonical",
        "--input-activity-csv-column",
        "molwt",
        "--input-task-csv-column",
        "one_taskid",
        "--input-initial-template",
        str(initial_config),
        "--input-retrain-template",
        str(retrain_config),
        "--input-slurm-template",
        str(slurm_template),
        "--n-cores",
        "1",
        "--dry-run",
        "-vv",
        "--slurm-al-pool",
        str(file_drd2_50),
        "--slurm-al-smiles-csv-column",
        "canonical",
        "--slurm-job-prefix",
        "testaml"
        "--slurm-req-partition",
        "testpartition"
    ]

    # Initiate a first run
    with patch.object(sys, "argv", automl_args):
        automl.main()
    ## check folders are created
    for f in ["output_path", "output_path/data"]:
        assert os.path.isdir(str(shared_datadir / f)), "output paths not created"
    ## check files are created
    for f in [
        "output_path/processed_timepoints.json",
        "output_path/data/TID1/TID1.csv",
        "output_path/data/TID1/TID1.sh",
        "output_path/data/TID1/TID1.json",
        "output_path/data/TID1/.24_01_01",
    ]:
        assert os.path.isfile(str(shared_datadir / f)), "output file not created"
    ## check timepoints initiated
    assert (
        json.load(
            open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
        )
        == []
    ), "AutoML not tracking completed timepoints"

    # Emulate scheduled job failure
    with patch.object(sys, "argv", automl_args):
        automl.main()
    df = pd.read_csv(str(shared_datadir / "output_path/data/TID1/.retry"))
    assert (
        df.loc[0][0] == 1
    ), "AutoML is not correctly locking and/or tracking retry attempts"
    assert (
        json.load(
            open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
        )
        == []
    ), "AutoML is not correctly tracking completed timepoints"
    assert os.path.isfile(str(shared_datadir / "output_path/data/TID1/.24_01_01")), "lock not still present"

    # Emulate the first has a trained model, but not yet unlocked
    ## Copy an example pkl to the path
    shutil.copyfile(drd2_pkl, str(shared_datadir / "output_path/data/TID1/TID.pkl"))
    with patch.object(sys, "argv", automl_args):
        automl.main()
    df = pd.read_csv(str(shared_datadir / "output_path/data/TID1/.retry"))
    assert (
        df.loc[0][0] == 2
    ), "AutoML is not correctly locking and/or tracking retry attempts"
    assert (
        json.load(
            open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
        )
        == []
    ), "AutoML is not correctly tracking completed timepoints"

    # Emulate a timepoint has successfully completed
    try:
        os.remove(str(shared_datadir / "output_path/data/TID1/.24_01_01"))
    except FileNotFoundError:
        pass
    with patch.object(sys, "argv", automl_args):
        automl.main()
    assert json.load(
        open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
    ) == ["24_01_01"], "AutoML is not correctly tracking completed timepoints"

    # Emulate a job is still running
    with patch.object(sys, "argv", automl_args):
        automl.main()
    assert json.load(
        open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
    ) == ["24_01_01"], "AutoML is not correctly waiting for unfinished jobs"

    # Ensure retries recounts from 0 with a new scheduled failed job
    os.remove(str(shared_datadir / "output_path/data/TID1/.retry"))
    with patch.object(sys, "argv", automl_args):
        automl.main()
    df = pd.read_csv(str(shared_datadir / "output_path/data/TID1/.retry"))
    assert df.loc[0][0] == 1, "AutoML is not tracking reset retry attempts"

    # Emulate the lock is removed and the erroneous dates 2024-04-01 (missing header),
    # 2024-04-01 (duplicate datapoints) and 2024-05-01 (all NaNs) are gracefully handled
    os.remove(str(shared_datadir / "output_path/data/TID1/.24_02_01"))
    with patch.object(sys, "argv", automl_args):
        automl.main()
    assert json.load(
        open(str(shared_datadir / "output_path/processed_timepoints.json"), "r")
    ) == [
        "24_01_01",
        "24_02_01",
        "24_03_01",
        "24_04_01",
        "24_05_01",
    ], "AutoML is not gracefully handling erroneous, duplicate and NaN timepoints correctly"
