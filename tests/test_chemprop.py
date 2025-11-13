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
        str(attach_root_path("examples/optimization/ChemProp_drd2_50.json")),
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


def test_optbuild_cli_si(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/ChemProp_drd2_50_sideinfo.json")),
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


def test_optbuild_si_cls(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path("examples/optimization/ChemProp_drd2_50_sideinfo_cls.json")
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


@pytest.mark.skip(reason="Await ChemProp fix for MPS RuntimeError")
def test_optbuild_pretrained_reg(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/ChemProp_drd2_50_retrain.json")),
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


def test_optbuild_pretrained_missing_file(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/ChemProp_drd2_50_retrain_missingfile.json"
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



def test_optbuild_pretrained_reg_on_cls(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/ChemProp_drd2_50_retrain_cls_error.json"
            )
        ),
        "--best-buildconfig-outpath",
        str(clean_shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(clean_shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(clean_shared_datadir / "merged.pkl"),
    ]

    with pytest.raises(
        ValueError,
        match="Exiting since no trials returned values",
    ):
        with patch.object(sys, "argv", testargs):
            optbuild.main()


def test_optbuild_cli_covariate(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/ChemProp_drd2_50_covariate.json")),
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

def test_optbuild_si_cls_covariate(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path("examples/optimization/ChemProp_drd2_50_sideinfo_cls_covariate.json")
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


def test_optbuild_peptide_classification_ChemProp(clean_shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path("examples/optimization/peptide_classification_ChemProp.json")
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