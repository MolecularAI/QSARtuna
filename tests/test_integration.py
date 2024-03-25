import json
import os
import sys
import tempfile
from unittest.mock import patch
import pandas as pd

import pytest
from apischema import serialize, deserialize

from optunaz import optbuild
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    LogisticRegression,
    SVR,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    Lasso,
    PLSRegression,
    XGBRegressor,
    SVC,
    AdaBoostClassifier,
    OptimizationConfig,
    ChemPropRegressor,
    ChemPropHyperoptRegressor,
    ChemPropClassifier,
    ChemPropHyperoptClassifier,
    PRFClassifier,
    CalibratedClassifierCVWithVA,
    Mapie,
)

from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    MACCS_keys,
    ECFP_counts,
    UnscaledPhyschemDescriptors,
    SmilesFromFile,
    SmilesAndSideInfoFromFile,
)
from optunaz import predict

from optunaz.utils.preprocessing.transform import (
    LogBase,
    LogNegative,
    VectorFromColumn,
    ZScales,
)
from optunaz.utils.files_paths import attach_root_path


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


@pytest.fixture
def file_drd2_50_err(shared_datadir):
    """Returns 50 molecules from DRD2 dataset, including some with errors."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train_with_errors.csv")


@pytest.fixture
def single_err_mol(shared_datadir):
    """Returns 1 err molecule for Jazzy descriptor"""
    return str(shared_datadir / "single_err_mol.csv")


@pytest.fixture
def file_drd2_50_side_info(shared_datadir):
    """Returns 50 molecules and side info from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train_side_info.csv")


@pytest.fixture
def file_sdf1(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "sdf" / "1.sdf")


@pytest.fixture
def train_with_fp(shared_datadir):
    """Returns sdf test file."""
    return str(shared_datadir / "precomputed_descriptor" / "train_with_fp.csv")


@pytest.fixture
def optconfig_regression(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            log_transform=True,
            log_transform_base=LogBase.LOG10,
            log_transform_negative=LogNegative.TRUE,
            log_transform_unit_conversion=6,
        ),
        descriptors=[ECFP.new(), UnscaledPhyschemDescriptors.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=15,
            random_seed=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression_sdf(file_sdf1):
    return OptimizationConfig(
        data=Dataset(
            input_column="Smiles",
            response_column="LogP",
            training_dataset_file=file_sdf1,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=10,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
        algorithms=[
            SVC.new(),
            RandomForestClassifier.new(
                n_estimators=RandomForestClassifier.Parameters.RandomForestClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
            LogisticRegression.new(),
            AdaBoostClassifier.new(
                n_estimators=AdaBoostClassifier.Parameters.AdaBoostClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=3,
            n_trials=10,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_calibration_rf(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=RandomForestClassifier.new(),
                n_folds=2,
                ensemble="True",
                method="vennabers",
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_calibration_chemprop(file_drd2_50, file_drd2_50_side_info):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            SmilesAndSideInfoFromFile.new(
                file=file_drd2_50_side_info,
                input_column="canonical",
                aux_weight_pc=SmilesAndSideInfoFromFile.Parameters.Aux_Weight_Pc(
                    low=2, high=10, q=2
                ),
            )
        ],
        algorithms=[
            CalibratedClassifierCVWithVA.new(
                estimator=ChemPropClassifier.new(
                    epochs=4,
                ),
                n_folds=2,
                ensemble="True",
                method="vennabers",
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            scoring="neg_brier_score",
        ),
    )


@pytest.fixture
def optconfig_mapie_rf(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            response_type="classification",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            Mapie.new(
                estimator=RandomForestRegressor.new(
                    n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                        low=2, high=2
                    )
                ),
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression_chemprop(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[SmilesFromFile.new()],
        algorithms=[
            ChemPropRegressor.new(
                epochs=4,
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_classification_chemprop(file_drd2_50, file_drd2_50_side_info):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt_gt_330",
            training_dataset_file=file_drd2_50,
        ),
        descriptors=[
            SmilesAndSideInfoFromFile.new(
                file=file_drd2_50_side_info,
                input_column="canonical",
            )
        ],
        algorithms=[
            ChemPropHyperoptClassifier.new(
                epochs=4,
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.CLASSIFICATION,
            cross_validation=2,
            n_trials=2,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_prfc(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_std=50,
            probabilistic_threshold_representation_threshold=330,
        ),
        descriptors=[ECFP.new()],
        algorithms=[
            PRFClassifier.new(
                n_estimators=PRFClassifier.Parameters.PRFClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize(
    "optconfig",
    [
        "optconfig_classification_chemprop",
        "optconfig_calibration_chemprop",
        "optconfig_calibration_chemprop",
        "optconfig_regression",
        "optconfig_mapie_rf",
        "optconfig_prfc",
        "optconfig_regression_sdf",
        "optconfig_classification",
        "optconfig_regression_chemprop",
        "optconfig_calibration_rf",
    ],
)
def test_qsartuna_integration(file_drd2_50, shared_datadir, optconfig, request):
    optconfig = request.getfixturevalue(optconfig)
    # optconfig.set_cache() Not setting cache to capture integration without cache test

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    for pkl_file in ["best.pkl", "merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


@pytest.fixture
def optconfig_mixture(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_std=50,
            probabilistic_threshold_representation_threshold=330,
        ),
        descriptors=[ECFP.new(), SmilesFromFile.new()],
        algorithms=[
            PRFClassifier.new(
                n_estimators=PRFClassifier.Parameters.PRFClassifierParametersNEstimators(
                    low=2, high=2
                )
            ),
            ChemPropHyperoptRegressor.new(epochs=4, num_iters=1),
            Mapie.new(
                estimator=RandomForestRegressor.new(
                    n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                        low=2, high=2
                    )
                ),
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=10,
            n_chemprop_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.mark.parametrize("split_chemprop", [True, False])
def test_mixture(file_drd2_50, shared_datadir, split_chemprop, optconfig_mixture):
    optconfig_mixture.settings.split_chemprop = split_chemprop
    optconfig_mixture.set_cache()
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_mixture)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


@pytest.fixture
def optconfig_mixture2(file_drd2_50, shared_datadir):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            training_dataset_file=file_drd2_50,
            probabilistic_threshold_representation=True,
            probabilistic_threshold_representation_std=50,
            probabilistic_threshold_representation_threshold=330,
        ),
        descriptors=[ECFP.new(), ECFP_counts.new(), SmilesFromFile.new()],
        algorithms=[
            Lasso.new(),
            PLSRegression.new(),
            ChemPropHyperoptRegressor.new(epochs=4, num_iters=1),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=2,
            n_trials=10,
            n_chemprop_trials=1,
            direction=OptimizationDirection.MAXIMIZATION,
            optuna_storage=f"sqlite:///{shared_datadir}/optuna_storage.sqlite",
            tracking_rest_endpoint="test",
            minimise_std_dev=True,
        ),
    )


@pytest.mark.parametrize("split_chemprop", [True, False])
def test_mixture2(file_drd2_50, shared_datadir, split_chemprop, optconfig_mixture2):
    optconfig_mixture2.settings.split_chemprop = split_chemprop
    optconfig_mixture2.set_cache()
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_mixture2)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        file_drd2_50,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 50


def test_physchem_integration(file_drd2_50, shared_datadir, optconfig_regression):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_regression)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".csv"
    ) as query:
        query.write("canonical\nC\nCC\nCC(=O)O[Hg]c1ccccc1\nCCC")

    predict_args = [
        "prog",
        "--model-file",
        str(shared_datadir / "best.pkl"),
        "--input-smiles-csv-file",
        query.name,
        "--input-smiles-csv-column",
        "canonical",
        "--output-prediction-csv-file",
        str(shared_datadir / "outprediction"),
    ]
    with patch.object(sys, "argv", predict_args):
        predict.main()

    predictions = pd.read_csv(
        str(shared_datadir / "outprediction"), usecols=["Prediction"]
    )
    assert len(predictions.dropna()) == 3
    assert any(predictions.loc[2].isna())


def test_err_jazzy(file_drd2_50_err, single_err_mol, shared_datadir):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        cfg = json.load(open(attach_root_path("examples/optimization/jazzy.json")))
        cfg["data"]["training_dataset_file"] = str(file_drd2_50_err)
        optconfig_fp.write(json.dumps(serialize(cfg)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".csv"
    ) as query:
        query.write(
            "canonical\nerr\nCOC(=O)Nc1cccc(c1)OCCCCN2CCC[C@H](C2)C(=O)N3[C@H]4C[C@H]3CN(C4)S(=O)(=O)C"
        )

    for f in [str(single_err_mol), query.name]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / "best.pkl"),
            "--input-smiles-csv-file",
            f,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 0


@pytest.fixture
def optconfig_regression_aux(file_drd2_50):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            aux_column="activity",
            training_dataset_file=file_drd2_50,
            log_transform=True,
            log_transform_base=LogBase.LOG10,
            log_transform_negative=LogNegative.TRUE,
            log_transform_unit_conversion=6,
        ),
        descriptors=[ECFP.new(), UnscaledPhyschemDescriptors.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=15,
            random_seed=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


@pytest.fixture
def optconfig_regression_vector_aux(train_with_fp):
    return OptimizationConfig(
        data=Dataset(
            input_column="canonical",
            response_column="molwt",
            response_type="regression",
            aux_column="fp",
            aux_transform=VectorFromColumn.new(),
            training_dataset_file=train_with_fp,
            log_transform=True,
            log_transform_base=LogBase.LOG10,
            log_transform_negative=LogNegative.TRUE,
            log_transform_unit_conversion=6,
        ),
        descriptors=[ECFP.new(), UnscaledPhyschemDescriptors.new(), MACCS_keys.new()],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(
                n_estimators=RandomForestRegressor.Parameters.RandomForestRegressorParametersNEstimators(
                    low=2, high=2
                )
            ),
            Ridge.new(),
            Lasso.new(),
            PLSRegression.new(),
            XGBRegressor.new(
                n_estimators=XGBRegressor.Parameters.NEstimators(low=2, high=2)
            ),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            cross_validation=3,
            n_trials=15,
            random_seed=1,
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    )


def test_regression_aux(file_drd2_50, optconfig_regression_aux, shared_datadir):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_regression_aux)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    for pkl_file in ["best.pkl", "merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--input-aux-column",
            "activity",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


def test_regression_vector_aux(
    train_with_fp, optconfig_regression_vector_aux, shared_datadir
):
    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig_regression_vector_aux)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    for pkl_file in ["best.pkl", "merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            train_with_fp,
            "--input-smiles-csv-column",
            "canonical",
            "--input-aux-column",
            "fp",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50


@pytest.mark.integration
@pytest.mark.parametrize(
    "optconfig",
    [
        "optconfig_regression",
        "optconfig_classification",
        "optconfig_regression_chemprop",
    ],
)
def test_scp_integration(file_drd2_50, shared_datadir, optconfig, request):
    optconfig = request.getfixturevalue(optconfig)

    with tempfile.NamedTemporaryFile(
        mode="wt", delete=False, dir=shared_datadir, suffix=".json"
    ) as optconfig_fp:
        optconfig_fp.write(json.dumps(serialize(optconfig)))

    testargs = [
        "prog",
        "--config",
        str(optconfig_fp.name),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
        "--inference_uncert",
        "None"
    ]
    with patch.object(sys, "argv", testargs):
        optbuild.main()

    os.unlink(optconfig_fp.name)

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None

    for pkl_file in ["best.pkl", "merged.pkl"]:
        predict_args = [
            "prog",
            "--model-file",
            str(shared_datadir / pkl_file),
            "--input-smiles-csv-file",
            file_drd2_50,
            "--input-smiles-csv-column",
            "canonical",
            "--output-prediction-csv-file",
            str(shared_datadir / "outprediction"),
        ]
        with patch.object(sys, "argv", predict_args):
            predict.main()

        predictions = pd.read_csv(
            str(shared_datadir / "outprediction"), usecols=["Prediction"]
        )
        assert len(predictions.dropna()) == 50
