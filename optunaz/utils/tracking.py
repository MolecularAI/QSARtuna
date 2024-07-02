import logging
import os
from dataclasses import dataclass
from typing import List, Dict

import requests
from apischema import serialize
from optunaz.config.build_from_opt import remove_algo_hash
from optuna import Study
from optuna.trial import FrozenTrial

from optunaz.config.build_from_opt import buildconfig_from_trial
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.evaluate import get_train_test_scores
from optunaz.model_writer import wrap_model

logger = logging.getLogger(__name__)


def get_authorization_header():
    return os.getenv("REINVENT_JWT")


@dataclass
class TrackingData:
    """Dataclass defining internal tracking format"""

    trial_number: int
    trial_value: float
    scoring: str
    trial_state: str
    all_cv_test_scores: Dict[str, List[float]]
    buildconfig: BuildConfig

    def __post_init__(self):
        self.buildconfig.metadata = None  # Metadata is not essential - drop.
        self.buildconfig.settings.n_trials = None  # Drop.


def removeprefix(line: str, prefix: str) -> str:
    # Starting from Python 3.9, str has method removeprefix().
    # We target Python 3.7+, so here is this function.
    if line.startswith(prefix):
        return line[len(prefix) :]


def round_scores(test_scores):
    return {k: [round(v, ndigits=3) for v in vs] for k, vs in test_scores.items()}


@dataclass
class InternalTrackingCallback:
    """Callback to track (log) progress using internal tracking format"""

    optconfig: OptimizationConfig
    trial_number_offset: int

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        trial = remove_algo_hash(trial)
        try:
            buildconfig = buildconfig_from_trial(study, trial)
            if hasattr(trial, "values") and trial.values is not None:
                trial_value = round(trial.values[0], ndigits=3)
            elif hasattr(trial, "value") and trial.value is not None:
                trial_value = round(trial.value, ndigits=3)
            else:
                trial_value = float("nan")

            data = TrackingData(
                trial_number=trial.number + self.trial_number_offset,
                trial_value=trial_value,
                scoring=self.optconfig.settings.scoring,
                trial_state=trial.state.name,
                all_cv_test_scores=round_scores(trial.user_attrs["test_scores"]),
                buildconfig=buildconfig,
            )

            json_data = serialize(data)

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": get_authorization_header(),
            }
            url = self.optconfig.settings.tracking_rest_endpoint
            try:
                response = requests.post(url, json=json_data, headers=headers)
            except Exception as e:
                logger.warning(f"Failed to report progress to {url}: {e}")
        except Exception as e:
            logger.warning(f"Failed to report progress: {e}")


@dataclass
class Datapoint:
    smiles: str
    expected: float
    predicted: float


@dataclass
class BuildTrackingData:
    """Dataclass defining internal tracking format"""

    response_column_name: str
    test_scores: Dict[str, float]
    test_points: List[Datapoint]


def track_build(model, buildconfig: BuildConfig):
    train_scores, test_scores = get_train_test_scores(model, buildconfig)

    rounded_test_scores = (
        {k: round(v, ndigits=3) for k, v in test_scores.items()}
        if test_scores is not None
        else None
    )

    _, _, _, smiles, expected, _ = buildconfig.data.get_sets()

    if smiles is None or len(smiles) < 1:
        logger.warning("No test set.")
        return

    mode = buildconfig.settings.mode
    descriptor = buildconfig.descriptor
    qsartuna_model = wrap_model(model, descriptor=descriptor, mode=mode)

    predicted = qsartuna_model.predict_from_smiles(smiles)

    test_points = [
        Datapoint(
            smiles=smi,
            expected=round(expval.item(), ndigits=3),  # item() converts numpy to float.
            predicted=round(predval.item(), ndigits=3),
        )
        for smi, expval, predval in zip(smiles, expected, predicted)
    ]

    data = BuildTrackingData(
        response_column_name=buildconfig.data.response_column,
        test_scores=rounded_test_scores,
        test_points=test_points,
    )

    json_data = serialize(data)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": get_authorization_header(),
    }
    url = buildconfig.settings.tracking_rest_endpoint

    try:
        response = requests.post(url, json=json_data, headers=headers)
    except Exception as e:
        logger.warning(f"Failed to report build results to {url}: {e}")
