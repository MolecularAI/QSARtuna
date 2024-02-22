import json
import sys
from unittest.mock import patch

from apischema import deserialize

from optunaz import optbuild
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils.files_paths import attach_root_path


def test_optbuild_cli(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/PRFC_P24863.json")),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None


def test_prf_eval(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/prf_comparison.json")),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None
