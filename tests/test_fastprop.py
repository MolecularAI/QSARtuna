import json
import sys
from unittest.mock import patch

import pytest
from apischema import deserialize

from optunaz import optbuild
from optunaz.config.buildconfig import BuildConfig
from optunaz.utils.files_paths import attach_root_path


def test_optbuild_cli(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/FastProp_drd2_50.json")),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None


def test_optbuild_cls(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/FastProp_drd2_50_cls.json")),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None


def test_optbuild_cli_covariate(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/FastProp_drd2_50_covariate.json")),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None


def test_optbuild_cls_covariate(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/FastProp_drd2_50_cls_covariate.json"
            )
        ),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    with open(clean_shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None
