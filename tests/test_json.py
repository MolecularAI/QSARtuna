import tempfile
import pytest
from apischema import deserialize
import optunaz.three_step_opt_build_merge
from optunaz.config.optconfig import OptimizationConfig

@pytest.mark.skip(reason="Not currently used")
def test(shared_datadir):
    data = """
    {
  "name": "15_09_06",
  "description": "",
  "data": {
    "training_dataset_file": "failed.csv",
    "input_column": "smiles",
    "response_column": "activity",
    "response_type": "classification",
    "deduplication_strategy": {
      "name": "KeepMedian"
    },
    "split_strategy": {
      "name": "ScaffoldSplit",
      "make_scaffold_generic": true,
      "butina_cluster": 0.4
    },
    "save_intermediate_files": false,
    "probabilistic_threshold_representation": false,
    "probabilistic_threshold_representation_threshold": null,
    "probabilistic_threshold_representation_std": null
  },
  "mode": "classification",
  "algorithms": [
    {
      "name": "RandomForestClassifier",
      "parameters": {
        "n_estimators": {
          "low": 100,
          "high": 250
        },
        "max_features": [
          "auto",
          "sqrt",
          "log2"
        ]
        }
    }
  ],
  "descriptors": [
    {
      "parameters": {
        "descriptors": [
          {
            "name": "ECFP",
            "parameters": {
              "radius": 3,
              "nBits": 2048
            }
          },
          {
            "name": "UnscaledPhyschemDescriptors",
            "parameters": {
            }
          },
          {
            "name": "UnscaledJazzyDescriptors",
            "parameters": {
            }
          }
        ]
      },
      "name": "CompositeDescriptor"
    }
  ],
  "settings": {
    "mode": "classification",
    "cross_validation": 5,
    "cv_split_strategy": {
      "name": "Stratified",
      "fraction": 0.2,
      "seed": 1
    },
    "shuffle": true,
    "direction": "maximize",
    "scoring": "roc_auc",
    "n_trials": 1,
    "n_jobs": -1,
    "n_startup_trials": 0,
    "track_to_mlflow": false,
    "tracking_rest_endpoint": null,
    "split_chemprop": false,
    "n_chemprop_trials": 0
  },
  "visualization": null,
  "task": "optimization"
}
    """
    import json

    config = deserialize(OptimizationConfig, json.loads(data.replace('failed.csv',str(shared_datadir / "failed.csv"))), additional_properties=True)
    config.data.training_dataset_file = str(
        shared_datadir / "failed.csv"
    )
    with tempfile.NamedTemporaryFile() as f:
        optunaz.three_step_opt_build_merge.optimize(config, f.name)