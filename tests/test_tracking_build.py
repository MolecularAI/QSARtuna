import tempfile

import pytest
from apischema import deserialize

import optunaz.three_step_opt_build_merge
from optunaz.config import ModelMode
from optunaz.config.buildconfig import (
    BuildConfig,
    Lasso,
    RandomForestClassifier,
    MapieRegressor, CatBoostRegressor,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP
from optunaz.utils.preprocessing.splitter import Random
from optunaz.utils.preprocessing.transform import (
    LogBase,
    LogNegative,
)


@pytest.fixture
def file_drd2_50(clean_shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(clean_shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def buildconfig_regression(file_drd2_50, clean_shared_datadir):
    return BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
            test_dataset_file=str(clean_shared_datadir / "DRD2" / "subset-50" / "test.csv"),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=Lasso.new(),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


@pytest.fixture
def buildconfig_mapie(file_drd2_50, clean_shared_datadir):
    return BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
            test_dataset_file=str(clean_shared_datadir / "DRD2" / "subset-50" / "test.csv"),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=MapieRegressor.new(
            estimator=Lasso.new(),
            n_folds=2,
            mapie_alpha=0.05,
            test_size=0.2,
            random_state=42,
        ),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )


def test_build_tracking(clean_shared_datadir, buildconfig_regression):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(
            buildconfig_regression, build_best_pkl.name
        )


def test_build_mapie(clean_shared_datadir, buildconfig_mapie):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(
            buildconfig_mapie, build_best_pkl.name
        )


def test_build_notest(clean_shared_datadir):
    buildconfig = BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
            split_strategy=Random(fraction=0.2),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=CatBoostRegressor.new(),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)


def test_build_cls(clean_shared_datadir):
    buildconfig = BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
            split_strategy=Random(fraction=0.2),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=RandomForestClassifier.new(max_features="auto"),
        settings=BuildConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)


def test_3(clean_shared_datadir):
    data = """
    {
    "data":
    {
        "training_dataset_file": "tests/data/DRD2/subset-50/train.csv",
        "input_column": "canonical",
        "response_column": "molwt",
        "test_dataset_file": null,
        "deduplication_strategy":
        {
            "name": "KeepAllNoDeduplication"
        },
        "split_strategy":
        {
            "name": "Temporal",
            "fraction": 0.2
        },
        "save_intermediate_files": true,
        "intermediate_training_dataset_file": "../intermediate_training_dataset_file.csv"
    },
    "metadata": null,
    "descriptor":
    {
        "name": "ECFP_counts",
        "parameters":
        {
            "radius": 3,
            "useFeatures": true,
            "nBits": 2048
        }
    },
    "settings":
    {
        "mode": "regression",
        "scoring": "r2",
        "direction": null,
        "n_trials": null,
        "tracking_rest_endpoint": "http://localhost:8891"
    },
    "algorithm":
    {
        "name": "PLSRegression",
        "parameters":
        {
            "n_components": 3
        }
    },
    "task": "building",
    "name": "Build from trial #47",
    "description": "Build from trial #47"
}
    """
    import json

    buildconfig = deserialize(BuildConfig, json.loads(data), additional_properties=True)
    buildconfig.data.training_dataset_file = str(
        clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
    )
    buildconfig.data.intermediate_test_dataset_file = str(
        clean_shared_datadir / "int_test.csv"
    )
    buildconfig.data.intermediate_training_dataset_file = str(
        clean_shared_datadir / "int_train.csv"
    )
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)


def test_4(clean_shared_datadir):
    data = """
    {
    "data":
    {
        "training_dataset_file": "tests/data/DRD2/subset-50/train.csv",
        "input_column": "canonical",
        "response_column": "molwt",
        "test_dataset_file": null,
        "deduplication_strategy":
        {
            "name": "KeepAllNoDeduplication"
        },
        "split_strategy":
        {
            "name": "Temporal",
            "fraction": 0.2
        },
        "save_intermediate_files": true,
        "intermediate_training_dataset_file": "../intermediate_training_dataset_file.csv"
    },
    "metadata": null,
    "descriptor":
    {
        "name": "ECFP_counts",
        "parameters":
        {
            "radius": 3,
            "useFeatures": true,
            "nBits": 2048
        }
    },
    "settings":
    {
        "mode": "regression",
        "scoring": "r2",
        "direction": null,
        "n_trials": null,
        "tracking_rest_endpoint": "http://localhost:8891"
    },
    "algorithm":
    {
        "name": "PLSRegression",
        "parameters":
        {
            "n_components": 3
        }
    },
    "task": "building",
    "name": "Build from trial #47",
    "description": "Build from trial #47"
}
    """
    import json

    buildconfig = deserialize(BuildConfig, json.loads(data), additional_properties=True)
    buildconfig.data.training_dataset_file = str(
        clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
    )
    buildconfig.data.intermediate_test_dataset_file = str(
        clean_shared_datadir / "int_test.csv"
    )
    buildconfig.data.intermediate_training_dataset_file = str(
        clean_shared_datadir / "int_train.csv"
    )
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)


def test_build_notestset(clean_shared_datadir):
    buildconfig = BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "DRD2" / "subset-50" / "train.csv"
            ),
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=Lasso.new(),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)


def test_build_aux(clean_shared_datadir):
    buildconfig = BuildConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            training_dataset_file=str(
                clean_shared_datadir / "aux_descriptors_datasets" / "train_with_conc.csv"
            ),
            test_dataset_file=str(
                clean_shared_datadir / "aux_descriptors_datasets" / "train_with_conc.csv"
            ),
            covariate_column="aux2",
            log_transform=True,
            log_transform_base=LogBase.LOG10,
            log_transform_negative=LogNegative.FALSE,
            log_transform_unit_conversion=2,
        ),
        metadata=None,
        descriptor=ECFP.new(),
        algorithm=Lasso.new(),
        settings=BuildConfig.Settings(
            mode=ModelMode.REGRESSION,
            tracking_rest_endpoint="http://localhost:8891",  # To listen: nc -l -k 8891
        ),
    )
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=clean_shared_datadir, suffix=".pkl"
    ) as build_best_pkl:
        optunaz.three_step_opt_build_merge.build_best(buildconfig, build_best_pkl.name)
