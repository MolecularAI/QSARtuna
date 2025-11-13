import tempfile
import pytest
from apischema import deserialize
import optunaz.three_step_opt_build_merge
from optunaz.config.optconfig import OptimizationConfig


@pytest.mark.skip(reason="Not currently used")
def test(clean_shared_datadir):
    data = """{
    "data": {
        "input_column": "Structure",
        "log_transform": false,
        "split_strategy": {
            "name": "Predefined",
            "column_name": "NN based;RH Clint Training/Prospective_LABEL"
        },
        "response_column": "ST00027 (Rat Heps Met Clint);GMean;CLint (µl/min/1E6)",
        "log_transform_base": "log10",
        "training_dataset_file": "failed.csv",
        "deduplication_strategy": {"name": "KeepAllNoDeduplication"},
        "log_transform_negative": "False",
        "log_transform_unit_conversion": null,
        "probabilistic_threshold_representation": false,
        "probabilistic_threshold_representation_std": 0,
        "probabilistic_threshold_representation_threshold": 0
    },
    "name": "DPR1 RH CLINT STALK SIDEINFO CHEMPROP",
    "task": "optimization",
    "settings": {
        "n_jobs": -1,
        "n_trials": 10,
        "random_seed": 1,
        "split_chemprop": false,
        "track_to_mlflow": false,
        "n_splits": 5,
        "minimise_std_dev": false,
        "n_startup_trials": 15,
        "optimization_split_strategy": {"name": "Random", "seed": 1, "fraction": 0.2},
        "n_chemprop_trials": 1
    },
    "algorithms": [
        {
            "name": "ChemPropRegressor",
            "parameters": {
                "depth": {"q": 1, "low": 2, "high": 6},
                "epochs": 40,
                "dropout": {"q": 0.04, "low": 0, "high": 1},
                "activation": ["ReLU"],
                "batch_size": {"q": 5, "low": 500, "high": 500},
                "max_lr_exp": {"low": -3, "high": -3},
                "aggregation": ["mean"],
                "message_hidden_dim": {"q": 100, "low": 300, "high": 2400},
                "ffn_num_layers": {"q": 1, "low": 1, "high": 3},
                "ffn_hidden_dim": {"q": 100, "low": 300, "high": 2400},
                "aggregation_norm": {"q": 1, "low": 100, "high": 100},
                "init_lr_ratio_exp": {"low": -4, "high": -4},
                "molecule_featurizers": ["none", "morgan_count", "rdkit_2d_normalized"],
                "final_lr_ratio_exp": {"low": -4, "high": -4},
                "warmup_epochs_ratio": {"q": 0.1, "low": 0.1, "high": 0.1}
            }
        }
    ],
    "description": "",
    "descriptors": [
        {"name": "SmilesFromFile"},
        {
            "name": "SmilesAndSideInfoFromFile",
            "parameters": {
                "file": "failed_si.csv",
                "input_column": "Structure"
            }
        }
    ],
    "slurmOptions": {
        "gres": "gpu:1",
        "nodes": 1,
        "ntasks": 1,
        "timeout": "100:0:0",
        "memPerCpu": "4G",
        "partition": "gpu",
        "buildMerged": false,
        "cpusPerTask": 6,
        "reservation": true,
        "saveAsSciKit": false,
        "optimizationJobId": null
    },
    "shuffle": true,
    "leave_out": 0.94,
    "n_trials": 50,
    "n_startup_trials": 10,
    "n_jobs": -1,
    "minimise_std_dev": true
  },
  "task": "optimization"
}
    """
    import json

    config = deserialize(
        OptimizationConfig,
        json.loads(
            data.replace("failed.csv", str(clean_shared_datadir / "failed.csv")).replace(
                "failed_si.csv", str(clean_shared_datadir / "failed_si.csv")
            )
        ),
        additional_properties=True,
    )
    config.data.training_dataset_file = str(clean_shared_datadir / "failed.csv")
    with tempfile.NamedTemporaryFile() as f:
        optunaz.three_step_opt_build_merge.optimize(config, f.name)
