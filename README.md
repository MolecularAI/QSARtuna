# QSARTUNA: QSAR Modeling with ML and Hyperparameter Tuning

Build predictive QSAR models with hyperparameters optimized using [Optuna](https://optuna.org/).

## Installation (for developers)

First clone the repository using Git.

Then execute the following commands in the root of the repository 

    conda env create -f env-dev.yml
    conda activate qsartuna
    poetry install --all-extras

the `qsartuna` package is now installed in editable mode.

## Background

This library searches for the best ML algorithm and molecular descriptor for the given data.

### The three-step process

QSARTUNA is structured around three steps:

1. *Hyperparameter Optimization:* 
    Train many models with different parameters using Optuna.
    Only the training dataset is used here. 
    Training is usually done with cross-validation.

2. *Build (Training):* 
    Pick the best model from Optimization, 
    and optionally evaluate its performance on the test dataset.

3. *"Prod-build:"* 
    Re-train the best-performing model on the merged training and test datasets. 
    This step has a drawback that there is no data left to evaluate the resulting model, 
    but it has a big benefit that this final model is trained on the all available data.   


## JSON-based Command-line interface on AZ SCP

Let's look at a trivial example of modelling molecular weight
using a training set of 50 molecules.

### Configuration file

We start with a configuration file in [JSON format](https://en.wikipedia.org/wiki/JSON).
It contains four main sections:
* **data** - location of the data file, columns to use.
* **settings** - details about the optimization run.
* **descriptors** - which molecular descriptors to use.
* **algorithms** - which ML algorithms to use.

Below is the example of such file 
(it is also available in examples/optimization/regression.json):

```json
{
  "task": "optimization",
  "data": {
    "training_dataset_file": "tests/data/DRD2/subset-50/train.csv",
    "input_column": "canonical",
    "response_column": "molwt"
  },
  "settings": {
    "mode": "regression",
    "n_splits": 5,
    "direction": "maximize",
    "n_trials": 100,
    "n_startup_trials": 30
  },
  "descriptors": [
    {
      "name": "ECFP",
      "parameters": {
        "radius": 3,
        "nBits": 2048
      }
    },
    {
      "name": "MACCS_keys",
      "parameters": {}
    }
  ],
  "algorithms": [
    {
      "name": "RandomForestRegressor",
      "parameters": {
        "max_depth": {"low": 2, "high": 32},
        "n_estimators": {"low": 10, "high": 250},
        "max_features": ["auto"]
      }
    },
    {
      "name": "Ridge",
      "parameters": {
        "alpha": {"low": 0, "high": 2}
      }
    },
    {
      "name": "Lasso",
      "parameters": {
        "alpha": {"low": 0, "high": 2}
      }
    },
    {
      "name": "XGBRegressor",
      "parameters": {
        "max_depth": {"low": 2, "high": 32},
        "n_estimators": {"low": 3, "high": 100},
        "learning_rate": {"low": 0.1, "high": 0.1}
      }
    }
  ]
}
```

Data section specifies location of the dataset file.
In this example it specifies a relative path to the `tests/data` folder.

Settings section specifies that:
* we are building a regression model,
* we want to use 5-fold cross-validation,
* we want to maximize the value of the objective function (maximization is the standard for scikit-learn models),
* we want to have a total of 100 trials,
* and the first 30 trials ("startup trials") should be random exploration (to not get stuck early on in one local minimum).

We specify two descriptors and four algorithm,
and optimization is free to pair any specified descriptor with any of the algorithms.

When we have our data and our configuration, it is time to start the optimization.

## Running via CLI

QSARtuna can be deployed directly from the CLI with any of the following tools:
```shell
qsartuna-<optimize|build|predict|schemagen|automl|metadata|convert> <command>
```

Example of running three-step-process from command line with the following command:

```shell
  qsartuna-optimize \
  --config examples/optimization/regression_drd2_50.json \
  --best-buildconfig-outpath path/to/your/output-dir/best.json \
  --best-model-outpath path/to/your/output-dir/best.pkl \
  --merged-model-outpath path/to/your/output-dir/merged.pkl
```
See each script for information about input arguments.


## Run from Python/Jupyter Notebook

You can use QSARtuna inside a python script or notebook:
```python
from qsartuna.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from qsartuna.config import ModelMode, OptimizationDirection
from qsartuna.config.optconfig import (
    OptimizationConfig,
    SVR,
    Ridge,
    Lasso,
)
from qsartuna.datareader import Dataset
from qsartuna.descriptors import ECFP, MACCS_keys, ECFP_counts

# Prepare hyperparameter optimization configuration.
config = OptimizationConfig(
    data=Dataset(
        input_column="canonical",  # Typical names are "SMILES" and "smiles".
        response_column="molwt",  # Often a specific name (like here), or just "activity".
        training_dataset_file="../tests/data/DRD2/subset-50/train.csv",
    ),
    descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
    algorithms=[SVR.new(), Ridge.new(), Lasso.new()],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        n_trials=100,
        direction=OptimizationDirection.MAXIMIZATION,
    ),
)

##
# Run Optuna Study.
study = optimize(config, study_name="my_study")

##
# Get the best Trial from the Study and make a Build (Training) configuration for it.
buildconfig = buildconfig_best(study)
# Optional: write out JSON of the best configuration.
import json
print(json.dumps(buildconfig.json(), indent=2))

##
# Build (re-Train) and save the best model.
build_best(buildconfig, "path/to/your/output-dir/best.pkl")

##
# Build (Train) and save the model on the merged train+test data.
build_merged(buildconfig, "path/to/your/output-dir/merged.pkl")
```

### Adding descriptors to QSARtuna


Add the descriptor code to the optunaz.descriptor.py file like so:

```python
@dataclass
class YourNewDescriptor(RdkitDescriptor):
    """YOUR DESCRIPTION GOES HERE"""

    @apischema.type_name("YourNewDescriptorParams")
    @dataclass
    class Parameters:
        # Any parameters to pass to your descriptor here
        exampleOfAParameter: Annotated[
            int,
            schema(
                min=1,
                title="exampleOfAParameter",
                description="This is an example int parameter.",
            ),
        ] = field(
            default=1,
        )

    name: Literal["YourNewDescriptor"]
    parameters: Parameters

    def calculate_from_smi(self, smi: str):
        # Insert your code to calculate from SMILES here
        fp = code_to_calculate_fp(smi)
        return fp
```

Then add the descriptor to the list here:

```python
AnyUnscaledDescriptor = Union[
    Avalon,
    ECFP,
    ECFP_counts,
    PathFP,
    AmorProtDescriptors,
    MACCS_keys,
    PrecomputedDescriptorFromFile,
    UnscaledMAPC,
    UnscaledPhyschemDescriptors,
    UnscaledJazzyDescriptors,
    UnscaledZScalesDescriptors,
    YourNewDescriptor, #Ensure your new descriptor added here
]
```

and here:

```python
CompositeCompatibleDescriptor = Union[
    AnyUnscaledDescriptor,
    ScaledDescriptor,
    MAPC,
    PhyschemDescriptors,
    JazzyDescriptors,
    ZScalesDescriptors,
    YourNewDescriptor, #Ensure your new descriptor added here
]
```

The YourNewDescriptor is now available as a QSARTUNA descriptor:
```python
from qsartuna.descriptors import YourNewDescriptor

config = OptimizationConfig(
    data=Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file="tests/data/DRD2/subset-50/train.csv",
    ),
    descriptors=[YourNewDescriptor.new()],
    algorithms=[
        SVR.new(),
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        n_splits=3,
        n_trials=100,
        direction=OptimizationDirection.MAXIMIZATION,
    ),
)
```

or in a new config:

```json
{
  "task": "optimization",
  "data": {
    "training_dataset_file": "tests/data/DRD2/subset-50/train.csv",
    "input_column": "canonical",
    "response_column": "molwt"
  },
  "settings": {
    "mode": "regression",
    "n_splits": 5,
    "direction": "maximize",
    "n_trials": 100,
    "n_startup_trials": 30
  },
  "descriptors": [
    {
      "name": "YourNewDescriptor",
      "parameters": {
        "exampleOfAParameter": 3
      }
    }
  ],
  "algorithms": [
    {
      "name": "RandomForestRegressor",
      "parameters": {
        "max_depth": {"low": 2, "high": 32},
        "n_estimators": {"low": 10, "high": 250},
        "max_features": ["auto"]
      }
    }
  ]
}
```

## Converting models to QSARtuna models

QSARtuna has a CLI helper to convert models to qsartuna models

We can perform this with the following command:

```shell
  qsartuna-convert \
  --input-model-file path/to/your/input-dir/sklearn_model.pkl\
  --input-model-mode regression \
  --output-model-path path/to/your/output-dir/qsartuna.pkl \
```

The convert CLI tool accepts the following command line arguments:

```
shell
qsartuna-convert -h
usage: qsartuna-convert [-h] --input-model-file INPUT_MODEL_FILE --input-model-mode INPUT_MODEL_MODE --output-model-path OUTPUT_MODEL_PATH [--input-json-descriptor-file INPUT_JSON_DESCRIPTOR_FILE] [--wrap-for-uncertainty]

Convert an existing sklearn(-like) model into a qsartuna model

options:
  -h, --help            show this help message and exit
  --input-json-descriptor-file INPUT_JSON_DESCRIPTOR_FILE
                        Name of input JSON file with descriptor configuration. Defaults to PrecomputedDescriptorFromFile
  --wrap-for-uncertainty
                        Whether to wrap regression in MAPIE or classification in VennAbers Calibrated Classifiers for uncertainty support

required named arguments:
  --input-model-file INPUT_MODEL_FILE
                        Model file name.
  --input-model-mode INPUT_MODEL_MODE
                        Classification or regression mode for the existing model.
  --output-model-path OUTPUT_MODEL_PATH
                        Path where to write the converted model.
```

Advanced options for the QSARtuna tools are covered in the QSARtuna notebook tutorial.
