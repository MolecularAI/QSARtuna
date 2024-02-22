import tempfile

import pytest

import optunaz
import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode
from optunaz.config.buildconfig import BuildConfig, SVC
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP_counts


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_test(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "test.csv")


@pytest.fixture
def buildconfig_classification(file_drd2_50, file_drd2_50_test):
    return BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
            test_dataset_file=file_drd2_50_test,
        ),
        metadata=None,
        descriptor=ECFP_counts.new(),
        algorithm=SVC.new(),
        settings=BuildConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


def test_svc(buildconfig_classification):
    buildconfig = buildconfig_classification
    with tempfile.NamedTemporaryFile() as f:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, f.name)
