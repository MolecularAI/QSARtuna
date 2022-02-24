# QPTUNA: QSAR using Optimization for Hyper-parameter Tuning

Build predictive models for CompChem with hyper-parameters optimized by [Optuna](https://optuna.org/).

## Background

This library searches 
for the best ML algorithm
and molecular descriptor
for the given data.

The search itself 
is done using [Optuna](https://optuna.org/).

### The three-step process

QPTUNA is structured around three steps:

1. *Hyper-parameter Optimization:*
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

## JSON-based Command-line interface

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
(it is also available in the source code repository).:

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
    "cross_validation": 5,
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
In this example it specifies a relative path to the `tests/data` folder
in the source code repository.

Settings section specifies that:
* we are building a regression model,
* we want to use 5-fold cross-validation,
* we want to maximize the value of the objective function (maximization is the standard for scikit-learn models),
* we want to have a total of 100 trials,
* and the first 30 trials ("startup trials") should be random exploration (to not get stuck early on in one local minimum).

We specify two descriptors and four algorithm,
and optimization is free to pair any specified descriptor with any of the algorithms.

When we have our data and our configuration, it is time to start the optimization.

### Running from the command line

QPTUNA can be deployed on a cluster using [Singularity](https://sylabs.io/guides/3.7/user-guide/index.html) container.

The commands below assume the use of Singularity container.
If QPTUNA is installed locally, without containerization, 
just skip the container part.

To run commands inside the container, Singularity uses the following syntax:
```shell
singularity exec <container.sif> <command>
```

We can create environmental variable to hold path to our container.
Using Bash syntax:
```bash
export QPTUNA_CONTAINER=/path/to/qptuna.sif
```

We can run three-step-process from command line with the following command:

```shell
singularity exec ${QPTUNA_CONTAINER} \
  /opt/qptuna/.venv/bin/qptuna-optimize \
  --config examples/optimization/regression_drd2_50.json \
  --best-buildconfig-outpath ~/qptuna-target/best.json \
  --best-model-outpath ~/qptuna-target/best.pkl \
  --merged-model-outpath ~/qptuna-target/merged.pkl
```

Since optimization can be a long process,
we should avoid running it on the login node, 
and we should submit it to the SLURM queue instead.

### Submitting to SLURM

We can submit our script to the queue by giving `sbatch` the following script: 

```sh
#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --time=100:0:0

# This script illustrates how to run one configuration from Qptuna examples.
# The example we use is in examples/optimization/regression_drd2_50.json.

# The example we chose uses relative paths to data files, change directory.
cd /path/to/qptuna/examples/and/data

singularity exec \
   ${QPTUNA_CONTAINER} \
  /opt/qptuna/.venv/bin/qptuna-optimize \
  --config examples/optimization/regression_drd2_50.json \
  --best-buildconfig-outpath ~/qptuna-target/best.json \
  --best-model-outpath ~/qptuna-target/best.pkl \
  --merged-model-outpath ~/qptuna-target/merged.pkl
```

When the script is complete, it will create pickled model files inside your home directory under `~/qptuna-target/`.

### Using the model

When the model is built, run inference:
```shell
singularity exec  ${QPTUNA_CONTAINER} \
  /opt/qptuna/.venv/bin/qptuna-predict \
  --model-file target/merged.pkl \
  --input-smiles-csv-file tests/data/DRD2/subset-50/test.csv \
  --input-smiles-csv-column "canonical" \
  --output-prediction-csv-file target/prediction.csv
```

## Run from Python/Jupyter Notebook

Create conda environment with Jupyter and Install Qptuna there:
```shell
poetry build
python -m pip install dist/qptuna-2.4.2.tar.gz
```

Then you can use Qptuna inside your Notebook:
```python
from qptuna.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from qptuna.config import ModelMode, OptimizationDirection
from qptuna.config.optconfig import (
    OptimizationConfig,
    SVR,
    RandomForest,
    Ridge,
    Lasso,
    PLS,
    XGBregressor,
)
from qptuna.datareader import Dataset
from qptuna.descriptors import ECFP, MACCS_keys, ECFP_counts

##
# Prepare hyperparameter optimization configuration.
config = OptimizationConfig(
    data=Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file="tests/data/DRD2/subset-50/train.csv",
    ),
    descriptors=[ECFP.new(), ECFP_counts.new(), MACCS_keys.new()],
    algorithms=[
        SVR.new(),
        RandomForest.new(),
        Ridge.new(),
        Lasso.new(),
        PLS.new(),
        XGBregressor.new(),
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        cross_validation=3,
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
build_best(buildconfig, "target/best.pkl")

##
# Build (Train) and save the model on the merged train+test data.
build_merged(buildconfig, "target/merged.pkl")
```

### Generating plain Scikit-learn models (models compatible with current Reinvent)

Script `optbuild.py` has option `--model-persistence-mode` 
that you can set to `plain_sklearn`
to get models that has no dependency on OptunaAZ,
and that are compatible with current Reinvent GUI.

```shell
singularity exec  ${QPTUNA_CONTAINER} \
  /opt/qptuna/.venv/bin/qptuna-optimize \
  --config examples/optimization/regression_drd2_50.json \
  --best-model-outpath target/best-plain.pkl \
  --best-buildconfig-outpath target/best-plain.json \
  --model-persistence-mode plain_sklearn 
```

In Python, 
you can send `persist_as=ModelPersistenceMode.PLAIN_SKLEARN`
to `build_best()` and `build_merged()`.

## Contributors
- Alexey Voronov [@alexvoronov](https://github.com/AlexVoronov)
- Christian Margreitter [@cmargreitter](https://github.com/CMargreitter)
- Atanas Patronov [@patronov](https://github.com/Patronov)
