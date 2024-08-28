import json
import sys
from unittest.mock import patch

import pytest
from apischema import deserialize

from optunaz import optbuild
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils.files_paths import attach_root_path


def test_optbuild_cli(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/ChemProp_drd2_50.json")),
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


def test_optbuild_pretrained(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/ChemProp_drd2_50_retrain.json")),
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


def test_optbuild_pretrained_missing_file(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/ChemProp_drd2_50_retrain_missingfile.json"
            )
        ),
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


def test_optbuild_pretrained_reg_on_cls(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/ChemProp_drd2_50_retrain_cls_error.json"
            )
        ),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]

    with pytest.raises(
        ValueError,
        match="Exiting since no trials returned values",
    ):
        with patch.object(sys, "argv", testargs):
            optbuild.main()
