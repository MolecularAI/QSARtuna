import tempfile
import pytest
from apischema import deserialize
import optunaz.three_step_opt_build_merge
from optunaz.config.optconfig import OptimizationConfig
from optunaz.config.buildconfig import BuildConfig


def test(shared_datadir):
    data = """{
  "data": {
    "input_column": "Structure",
    "log_transform": true,
    "response_type": "regression",
    "split_strategy": {
      "name": "Predefined",
      "column_name": "NN based;RH Clint Training/Prospective_LABEL"
    },
    "response_column": "ST00027 (Rat Heps Met Clint);GMean;CLint (µl/min/1E6)",
    "test_dataset_file": null,
    "log_transform_base": "log10",
    "training_dataset_file": "failed.csv",
    "deduplication_strategy": {
      "name": "KeepAllNoDeduplication"
    },
    "log_transform_negative": "False",
    "log_transform_unit_conversion": 4,
    "probabilistic_threshold_representation": false,
    "probabilistic_threshold_representation_std": 0,
    "probabilistic_threshold_representation_threshold": 0
  },
  "name": "Build from trial #315",
  "task": "building",
  "metadata": null,
  "settings": {
    "mode": "regression",
    "scoring": "neg_mean_squared_error",
    "n_trials": null,
    "direction": "maximize"
  },
  "algorithm": {
    "name": "ChemPropRegressor",
    "parameters": {
      "depth": 6,
      "epochs": 4,
      "dropout": 0.16,
      "activation": "SELU",
      "batch_size": 500,
      "max_lr_exp": -3,
      "aggregation": "mean",
      "hidden_size": 1200,
      "aux_weight_pc": 100,
      "ensemble_size": 1,
      "ffn_num_layers": 2,
      "ffn_hidden_size": 700,
      "aggregation_norm": 100,
      "init_lr_ratio_exp": -4,
      "features_generator": "none",
      "final_lr_ratio_exp": -4,
      "warmup_epochs_ratio": 0.1
    }
  },
  "descriptor": {
    "name": "SmilesFromFile",
    "parameters": {}
  },
  "description": "Build from trial #315"
}
    """
    import json

    config = deserialize(
        BuildConfig,
        json.loads(
            data.replace("failed.csv", str(shared_datadir / "failed.csv")).replace(
                "failed_si.csv", str(shared_datadir / "failed_si.csv")
            )
        ),
        additional_properties=True,
    )
    config.data.training_dataset_file = str(shared_datadir / "failed.csv")
    with tempfile.NamedTemporaryFile() as f:
        optunaz.three_step_opt_build_merge.build(config, f.name)

