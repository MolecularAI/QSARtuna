import optuna
import pytest
from apischema import deserialize

from optunaz.config.optconfig import OptimizationConfig
from optunaz.objective import Objective
from optunaz.utils import load_json
from optunaz.utils.files_paths import attach_root_path


@pytest.mark.parametrize(
    "filepath",
    [
        "examples/optimization/RF_SVR_regression.json",
        "examples/optimization/Lasso_Ridge_regression.json",
    ],
)
def test_input_data_variants_1(shared_datadir, filepath):

    confdict = load_json.loadJSON(path=attach_root_path(filepath))
    confdict["data"]["training_dataset_file"] = str(
        shared_datadir / "DRD2/subset-50/train.csv"
    )
    confdict["data"]["test_dataset_file"] = str(
        shared_datadir / "DRD2/subset-50/test.csv"
    )
    confdict["data"]["input_column"] = "canonical"
    confdict["data"]["response_column"] = "molwt"

    conf = deserialize(OptimizationConfig, confdict)
    train_smiles, train_y, test_smiles, test_y = conf.data.get_sets()
    obj = Objective(optconfig=conf, train_smiles=train_smiles, train_y=train_y)
    study = optuna.create_study(direction=conf.settings.direction)
    study.optimize(obj, n_trials=1)
