import csv
import json
import os
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import numpy.testing as npt
from apischema import deserialize

import optunaz.three_step_opt_build_merge
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import PrecomputedDescriptorFromFile, ECFP
from optunaz.utils.preprocessing.transform import VectorFromColumn
from optunaz import optbuild
from optunaz.utils.files_paths import attach_root_path
from optunaz import predict


def test_1():
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["SMILES", "activity"])
        writer.writeheader()
        writer.writerow({"SMILES": "CCC", "activity": "0.5,-0.6,0.7"})
        writer.writerow({"SMILES": "CCCCC", "activity": "0.7, 0.8, 1"})

    d = PrecomputedDescriptorFromFile.new(
        file=f.name, input_column="SMILES", response_column="activity"
    )

    pred = d.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [0.5, -0.6, 0.7], 2)

    pred = d.calculate_from_smi("CCCCC")
    npt.assert_almost_equal(pred[0:3], [0.7, 0.8, 1.0], 2)

    os.unlink(f.name)


def test_from_existing_file(shared_datadir):
    file = shared_datadir / "precomputed_descriptor" / "tiny.csv"
    d = PrecomputedDescriptorFromFile.new(
        file=str(file), input_column="SMILES", response_column="activity"
    )

    pred = d.calculate_from_smi("CCC")
    npt.assert_almost_equal(pred[0:3], [0.5, -0.6, 0.7], 2)

    pred = d.calculate_from_smi("CCCCC")
    npt.assert_almost_equal(pred[0:3], [0.7, 0.8, 1.0], 2)


def smi2fp(smi):
    d = ECFP.new(nBits=512)
    return (
        np.array2string(
            d.calculate_from_smi(smi),
            formatter={"bool": lambda x: str(int(x))},
            max_line_width=10000,
            threshold=10000,
            separator=",",
        )
        .replace("[", "")
        .replace("]", "")
    )


def test_3(shared_datadir):
    ori_molfile = shared_datadir / "DRD2" / "subset-50" / "train.csv"
    df = pd.read_csv(ori_molfile)
    df["fp"] = df["canonical"].apply(smi2fp)
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)

    data = json.load(
        open(
            attach_root_path(
                "examples/optimization/regression_drd2_50_precomputed_descriptor.json"
            ),
            "rt",
        )
    )

    config = deserialize(OptimizationConfig, data)
    config.data.training_dataset_file = f.name
    config.descriptors[0].parameters.file = f.name

    optunaz.three_step_opt_build_merge.optimize(
        config, "test_regression_precomputed_descriptor"
    )

    os.unlink(f.name)


def test_4(shared_datadir):
    ori_molfile = shared_datadir / "DRD2" / "subset-50" / "train.csv"
    df = pd.read_csv(ori_molfile)
    df["fp"] = df["canonical"].apply(smi2fp)
    df["aux"] = df["canonical"].apply(smi2fp)
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)

    data = json.load(
        open(
            attach_root_path(
                "examples/optimization/regression_drd2_50_precomputed_descriptor.json"
            ),
            "rt",
        )
    )

    config = deserialize(OptimizationConfig, data)
    config.data.training_dataset_file = f.name
    config.descriptors[0].parameters.file = f.name
    config.data.aux_column = "aux"
    config.data.aux_transform = VectorFromColumn.new()

    optunaz.three_step_opt_build_merge.optimize(
        config, "test_precomputed_descriptor_auxcol"
    )

    os.unlink(f.name)


def test_5(shared_datadir):
    ori_molfile = shared_datadir / "DRD2" / "subset-50" / "train.csv"
    df = pd.read_csv(ori_molfile)
    df["aux"] = df["canonical"].apply(smi2fp)
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)

    data = json.load(
        open(
            attach_root_path("examples/optimization/ChemProp_drd2_50.json"),
            "rt",
        )
    )

    config = deserialize(OptimizationConfig, data)
    config.data.training_dataset_file = f.name
    config.descriptors[0].parameters.file = f.name
    config.data.aux_column = "aux"
    config.data.aux_transform = VectorFromColumn.new()

    optunaz.three_step_opt_build_merge.optimize(
        config, "test_precomputed_descriptor_auxcol_chemprop"
    )

    os.unlink(f.name)


def test_scaled_precomputed(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(
            attach_root_path(
                "examples/optimization/regression_drd2_50_precomputed_descriptor_scaled.json"
            )
        ),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        str(shared_datadir / "precomputed_descriptor/train_with_fp.csv"),
        "--input-smiles-csv-column",
        "canonical",
        "--input-precomputed-file",
        str(shared_datadir / "precomputed_descriptor/train_with_fp.csv"),
        "--input-precomputed-input-column",
        "canonical",
        "--input-precomputed-response-column",
        "fp",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        npt.assert_allclose(
            predictions.loc[[0, 1, 2]].values.flatten(),
            [385.872, 388.57, 379.173],
            rtol=1e-05,
            atol=1e-05,
        )
