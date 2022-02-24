import csv
import json
import os
import tempfile

import numpy.testing as npt
from apischema import deserialize

import optunaz.three_step_opt_build_merge
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import PrecomputedDescriptorFromFile, ECFP

import numpy as np
import pandas as pd

from optunaz.utils.files_paths import attach_root_path


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
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as f:
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

    data = json.load(
        open(
            attach_root_path(
                "examples/optimization/regression_drd2_50_precomputed_descriptor.json"
            ),
            "rt",
        )
    )
    config = deserialize(OptimizationConfig, data)

    molfile = str(shared_datadir / "precomputed_descriptor" / "train_with_fp.csv")
    config.data.training_dataset_file = molfile
    config.descriptors[0].parameters.file = molfile

    optunaz.three_step_opt_build_merge.optimize(
        config, "test_regression_precomputed_descriptor"
    )
