import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any

import mlflow
import optuna
from apischema import serialize
from mlflow.tracking import MlflowClient
from optuna.trial import FrozenTrial, TrialState

from optunaz.config.build_from_opt import buildconfig_from_trial
from optunaz.config.optconfig import OptimizationConfig
from optunaz.utils.enums import MlflowLogParams


def add_ellipsis(name: str, max_length=100) -> str:
    if len(name) > max_length - 3:
        return name[0 : (max_length - 3)] + "..."
    else:
        return name


def shorten_names(params: Dict[str, Any]) -> Dict[str, str]:
    return {add_ellipsis(str(k)): add_ellipsis(str(v)) for k, v in params.items()}


# This class is loosely based on experimental MLflow integration for Optuna:
# https://optuna.readthedocs.io/en/stable/_modules/optuna/integration/mlflow.html
@dataclass
class MLflowCallback:
    """Callback to track Optuna trials with MLflow.

    This callback adds Optuna-tracked information to MLflow.
    The MLflow experiment will be named after the Optuna study name.
    MLflow runs will be named after Optuna trial numbers.
    """

    trial_number_offset: int
    tracking_uri: Optional[str] = None
    """
    The URI of the MLflow tracking server.
    Please refer to `mlflow.set_tracking_uri
    <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>`_
    for more details.
    """

    optconfig: OptimizationConfig = None

    def __call__(self, study: optuna.study.Study, trial: FrozenTrial) -> None:
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(study.study_name)

        if hasattr(trial, "values") and trial.values is not None:
            trial_value = round(trial.values[0], ndigits=3)
        elif hasattr(trial, "value") and trial.value is not None:
            trial_value = round(trial.value, ndigits=3)
        else:
            trial_value = float("nan")

        with mlflow.start_run(
            run_name=str(trial.number + self.trial_number_offset)
        ) as run:
            metric_name = (
                f"optimization_objective_cvmean_{self.optconfig.settings.scoring}"
            )
            mlflow.log_metric(metric_name, trial_value)

            # Log individual scores from cross-validation iterations.
            # We (ab)use MLFlow 'step' parameter to log values from CV iterations.
            # Note: MLFlow UI displays value from the last iteration only.
            # Other iteration values can be seen in metric detail view.
            # Use MlflowClient().log_metric instead of mlflow.log_metric to log time.
            for metric_name, vals in trial.user_attrs["test_scores"].items():
                for iteration, value in enumerate(vals):
                    run_id = run.info.run_id
                    timestamp = int(trial.datetime_complete.timestamp() * 1000)
                    MlflowClient().log_metric(
                        run_id, metric_name, value, timestamp, iteration
                    )
                    # mlflow.log_metric(key=metric_name, value=v, step=i)

            mlflow.log_params(shorten_names(trial.params))
            # Log trial number as parameter, to use it in MLflow Compare Runs UI.
            mlflow.log_param(
                MlflowLogParams.TRIAL_NUMBER, trial.number + self.trial_number_offset
            )
            mlflow.set_tags(self.prepare_tags(study, trial))

            fname = self.tmp_buildconfig(study, trial)
            mlflow.log_artifact(fname)
            os.unlink(fname)

    def prepare_tags(
        self, study: optuna.study.Study, trial: FrozenTrial
    ) -> Dict[str, str]:
        """Sets the tags for MLflow."""

        tags: Dict[str, str] = {
            "number": str(trial.number + self.trial_number_offset),
            "datetime_start": str(trial.datetime_start),
            "datetime_complete": str(trial.datetime_complete),
        }

        # Set state and convert it to str and remove the common prefix.
        trial_state = trial.state
        if isinstance(trial_state, TrialState):
            tags["state"] = str(trial_state).split(".")[-1]

        # Set direction and convert it to str and remove the common prefix.
        if isinstance(study.direction, optuna.study.StudyDirection):
            tags["direction"] = str(study.direction).split(".")[-1]

        tags.update(trial.user_attrs)

        return tags

    def tmp_buildconfig(self, study, trial) -> str:
        """Creates a temporary file with build configuration, and returns a path to it."""

        buildconfig = buildconfig_from_trial(study, trial)
        with tempfile.NamedTemporaryFile(
            mode="wt", delete=False, prefix="buildconfig_", suffix=".json"
        ) as f:
            f.write(json.dumps(serialize(buildconfig)))

        return f.name
