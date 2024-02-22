import sys
import os
import json
import glob
import pytest
import tempfile
from unittest.mock import patch

from optunaz import optbuild
from apischema import deserialize, serialize
from optunaz.utils.files_paths import attach_root_path
from optunaz.config.optconfig import OptimizationConfig
from optunaz.config.buildconfig import BuildConfig


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_pxc50(shared_datadir):
    """Returns pxc50 dataset."""
    return str(shared_datadir / "pxc50" / "P24863.csv")


@pytest.mark.parametrize(
    "filename",
    glob.glob(f"{attach_root_path('examples/optimization_splitters')}/*.json"),
)
def test_optsplitters_cli(shared_datadir, filename, file_drd2_50, file_pxc50):
    data = json.load(open(filename))
    optconfig = deserialize(OptimizationConfig, data)
    if "pxc50" in optconfig.data.training_dataset_file:
        optconfig.data.training_dataset_file = file_pxc50
    else:
        optconfig.data.training_dataset_file = file_drd2_50

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None
