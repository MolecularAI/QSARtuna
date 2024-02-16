import pytest

from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepFirst, KeepAllNoDeduplication
from optunaz.utils.preprocessing.splitter import Random, Stratified


@pytest.fixture
def drd2_300(shared_datadir):
    return str(shared_datadir / "DRD2" / "subset-300" / "train.csv")


@pytest.fixture
def drd2_aux(shared_datadir):
    return str(shared_datadir / "aux_descriptors_datasets" / "train_with_conc.csv")


@pytest.fixture
def file_sdf1(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "sdf" / "1.sdf")

def test_nosplit(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 300
    assert len(test_smiles) == 0


def test_split(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60


def test_split_and_extra(drd2_300, shared_datadir):
    testfile = str(shared_datadir / "DRD2" / "subset-50" / "test.csv")
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        test_dataset_file=testfile,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60 + 50


def test_split_strat(drd2_300):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=drd2_300,
        deduplication_strategy=KeepFirst(),  # should not remove anything
        split_strategy=Stratified(fraction=0.2, seed=42),
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 240
    assert len(test_smiles) == 60


def test_sdf(file_sdf1):
    data = Dataset(
        input_column="Smiles",
        response_column="LogP",
        response_type="regression",
        training_dataset_file=file_sdf1,
        deduplication_strategy=KeepFirst(),
        split_strategy=Random(fraction=0.2, seed=42),
    )
    train_smiles, train_y, _, test_smiles, test_y, _ = data.get_sets()
    assert len(train_smiles) == len(train_y)
    assert len(test_smiles) == len(test_y)
    assert len(train_smiles) == 407
    assert len(test_smiles) == 102

@pytest.fixture
def drd2_reg_errs(shared_datadir):
    return str(
        shared_datadir / "DRD2" / "subset-50" / "train_response_col_issue_test.csv"
    )


def test_reg_errs(drd2_reg_errs):
    data = Dataset(
        input_column="canonical",
        response_column="molwt",
        response_type="regression",
        training_dataset_file=drd2_reg_errs,
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    # check that the ValueError is thrown for erroneous input file
    with pytest.raises(ValueError):
        try:
            data.get_sets()
        except ValueError as e:
            # check the number of detected erroneous lines is 6
            assert len(e.args[1]) == 6
            raise e


@pytest.fixture
def drd2_cls_errs(shared_datadir):
    return str(
        shared_datadir / "DRD2" / "subset-50" / "train_response_col_issue_test.csv"
    )


def test_cls_errs(drd2_cls_errs):
    data = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        response_type="classification",
        training_dataset_file=drd2_cls_errs,
        deduplication_strategy=KeepAllNoDeduplication(),
    )
    # check that the ValueError is thrown for erroneous input file
    with pytest.raises(ValueError):
        try:
            data.get_sets()
        except ValueError as e:
            # check the number of detected erroneous lines is 3
            assert len(e.args[1]) == 3
            raise e


@pytest.mark.parametrize(
    "aux_column,exp_tr,exp_te",
    [
        ("aux1", 40, 10),
        ("aux2", 80, 20),
        ("aux3", 80, 20),
    ],
)
def test_aux(drd2_aux, aux_column, exp_tr, exp_te):
    data = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        response_type="classification",
        aux_column=aux_column,
        training_dataset_file=drd2_aux,
        deduplication_strategy=KeepFirst(),
        split_strategy=Random(fraction=0.2, seed=42),
    )

    train_smiles, train_y, train_aux, test_smiles, test_y, test_aux = data.get_sets()
    assert len(train_smiles) == exp_tr
    assert len(test_smiles) == exp_te
    assert len(train_smiles) == exp_tr
    assert len(test_smiles) == exp_te
