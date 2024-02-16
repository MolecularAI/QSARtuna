import numpy as np
import pytest
import pandas as pd
from optunaz.utils.preprocessing.splitter import *
from optunaz.datareader import split as datareader_split
from optunaz.datareader import deduplicate
from optunaz.datareader import read_data
from rdkit import Chem
from optunaz.utils.preprocessing.deduplicator import *


@pytest.fixture
def drd2_50(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def drd2_100(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-100" / "train.csv")


@pytest.fixture()
def dataset():
    df = pd.DataFrame(
        {
            "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "val": [3, 10, 1, 7, 4],
        }
    )
    return df


@pytest.fixture
def hist_file(shared_datadir):
    return str(shared_datadir / "histogramsplit_check.csv")


def test_random(dataset):
    df = dataset

    train, test = Random(fraction=0.2, seed=42).split(df["SMILES"])
    assert len(df.iloc[train]) == 4
    assert len(df.iloc[test]) == 1
    assert df.iloc[test].iloc[0, 1] == 10

    train, test = Random(fraction=0.2, seed=42).split(df["SMILES"], df["val"])
    assert len(df.iloc[train]) == 4
    assert len(df.iloc[test]) == 1
    assert df.iloc[test].iloc[0, 1] == 10

    train, test = Random(fraction=0.4, seed=74).split(df["SMILES"])
    assert len(df.iloc[train]) == 3
    assert len(df.iloc[test]) == 2
    assert df.iloc[test].iloc[0, 1] == 1


def test_temporal(dataset):
    train, test = Temporal(fraction=0.4).split(dataset["SMILES"])
    assert len(train) == 3
    assert len(test) == 2
    assert dataset.iloc[test]["val"].tolist() == [7, 4]


def test_stratified():
    # Stratified requires at least 10 observations per bin,
    # generate bigger dataset.
    n = 30
    smiles = ["C" * i for i in range(1, n + 1)]
    np.random.seed(42)
    val = np.random.normal(size=n)
    dataset = pd.DataFrame({"SMILES": smiles, "val": val})

    train, test = Stratified(fraction=0.4, seed=42).split(
        dataset["SMILES"], dataset["val"]
    )
    assert len(train) == 18
    assert len(test) == 12


def test_stratified_firstbin_lessthan10(hist_file):
    # Checks edge cases treated correctly (i.e. first bin has fewer than 10 instances)
    dataset = pd.read_csv(hist_file)
    dataset["SMILES"] = "C"

    train, test = Stratified(fraction=0.2, seed=1).split(
        dataset["SMILES"], dataset["val"]
    )
    assert len(train) == 1297
    assert len(test) == 325


@pytest.mark.parametrize(
    "cutoff,tr,te",
    [
        (0.0, 80, 20),
        (0.2, 78, 22),
        (0.4, 95, 5),
        (0.6, 44, 56),
        (0.8, 4, 96),
        (1.0, 100, 0),
    ],
)
def test_scaffold(drd2_100, cutoff, te, tr):
    # test to increase the distance threshold for scaffold clustering
    dataset = pd.read_csv(drd2_100)
    sp = ScaffoldSplit(make_scaffold_generic=True, butina_cluster=cutoff)
    scaffolds = sp.groups(dataset, "canonical")
    assert len(scaffolds) == 100
    assert len(set(scaffolds.values())) == 94
    dataset["scaf"] = dataset["canonical"].map(scaffolds)
    train, test = sp.split(
        dataset["canonical"], dataset["molwt"], groups=dataset["scaf"]
    )
    assert len(train) == tr
    assert len(test) == te


@pytest.mark.parametrize(
    "dedup",
    [
        KeepFirst,
        KeepLast,
        KeepRandom,
        KeepAvg,
        KeepMedian,
        KeepMin,
        KeepMax,
        KeepAllNoDeduplication,
    ],
)
@pytest.mark.parametrize(
    "splitter",
    [
        Stratified(fraction=0.4, seed=42),
        Random(fraction=0.4),
        Temporal(fraction=0.4),
        ScaffoldSplit(),
    ],
)
def test_reg_duplicates(shared_datadir, drd2_50, dedup, splitter):
    train_smiles, train_y, _, train_groups = read_data(
        drd2_50,
        "canonical",
        "molwt",
        "regression",
        None,
        splitter,
    )
    smis = []
    groups = []
    for idx, smi in enumerate(train_smiles):
        mol = Chem.MolFromSmiles(smi)
        for _ in range(15):
            # doRandom randomises DFS transversal graph to generate random smiles
            smis.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            if train_groups is not None:
                groups.append(train_groups[idx])
    # create random vals and deduplicate using datareader's deduplicator
    np.random.seed(42)
    y = np.random.normal(size=len(smis))
    if train_groups is None:
        dataset = pd.DataFrame({"SMILES": smis, "val": y, "groups": train_groups})
    else:
        dataset = pd.DataFrame({"SMILES": smis, "val": y, "groups": groups})
    X = dataset["SMILES"]
    y = dataset["val"]
    groups = dataset["groups"]
    train_smiles, train_y, _, train_groups = deduplicate(
        X, y, None, groups, dedup(), "regression"
    )
    # split using datareader's splitter
    train_smiles, train_y, _, extra_test_smiles, extra_test_y, _ = datareader_split(
        train_smiles, train_y, None, splitter, train_groups
    )
    assert set(train_y.unique()) != {False, True}
    if train_groups is None:
        if "KeepAll" in str(dedup):
            tr = 450
            te = 300
        else:
            tr = 30
            te = 20
    elif "Scaffold" not in str(splitter):
        if "KeepAll" in str(dedup):
            tr = 600
            te = 150
        else:
            tr = 40
            te = 10
    # do not assert scaffold split sizes in this test
    else:
        return
    assert len(train_smiles) == tr
    assert len(train_y) == tr
    assert len(extra_test_smiles) == te
    assert len(extra_test_y) == te


@pytest.mark.parametrize(
    "dedup",
    [
        KeepAvg,
        KeepMedian,
        KeepAllNoDeduplication,
    ],
)
@pytest.mark.parametrize(
    "splitter",
    [
        Stratified(fraction=0.4, seed=42),
    ],
)
def test_cls_duplicates(shared_datadir, drd2_50, dedup, splitter):
    train_smiles, train_y, _, train_groups = read_data(
        drd2_50,
        "canonical",
        "molwt_gt_330",
        "classification",
        None,
        splitter,
    )
    smis = []
    groups = []
    for idx, smi in enumerate(train_smiles):
        mol = Chem.MolFromSmiles(smi)
        for _ in range(15):
            # doRandom randomises DFS transversal graph to generate random smiles
            smis.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            if train_groups is not None:
                groups.append(train_groups[idx])
    # create random vals and deduplicate using datareader's deduplicator
    np.random.seed(42)
    y = np.random.normal(size=len(smis)) > 0
    if train_groups is None:
        dataset = pd.DataFrame({"SMILES": smis, "val": y, "groups": train_groups})
    else:
        dataset = pd.DataFrame({"SMILES": smis, "val": y, "groups": groups})
    X = dataset["SMILES"]
    y = dataset["val"]
    groups = dataset["groups"]
    train_smiles, train_y, _, train_groups = deduplicate(
        X, y, None, groups, dedup(), "classification"
    )
    # split using datareader's splitter
    train_smiles, train_y, _, extra_test_smiles, extra_test_y, _ = datareader_split(
        train_smiles, train_y, None, splitter, train_groups
    )
    assert set(train_y.unique()) == {False, True}
    if train_groups is None:
        if "KeepAll" in str(dedup):
            tr = 450
            te = 300
        else:
            tr = 30
            te = 20
    elif "Scaffold" not in str(splitter):
        if "KeepAll" in str(dedup):
            tr = 600
            te = 150
        else:
            tr = 40
            te = 10
    # do not assert scaffold split sizes in this test
    else:
        return
    assert len(train_smiles) == tr
    assert len(train_y) == tr
    assert len(extra_test_smiles) == te
    assert len(extra_test_y) == te
