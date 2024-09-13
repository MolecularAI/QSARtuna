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
from optunaz.evaluate import calibration_analysis

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
    algorith_hash: str

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
    """Callback to track (log) Optimization progress using internal tracking format"""

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
                algorith_hash=trial.user_attrs["alg_hash"],
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
class Calpoint:
    bin_edges: float
    frac_true: float
    frac_pred: float


@dataclass
class BuildTrackingData:
    """Dataclass defining internal Build tracking format"""

    response_column_name: str
    test_scores: Dict[str, float] | str
    test_points: List[Datapoint]
    cal_points: List[Calpoint] | None


def track_build(qptuna_model, buildconfig: BuildConfig, test_scores):
    test_smiles = qptuna_model.predictor.test_smiles_
    test_aux = qptuna_model.predictor.test_aux_
    expected = qptuna_model.predictor.test_y_

    if test_smiles is None or len(test_smiles) < 1:
        logger.warning("No test set.")
        return
    rounded_test_scores = (
        {k: round(v, ndigits=3) for k, v in test_scores.items()}
        if test_scores is not None
        else ""
    )

    predicted = qptuna_model.predict_from_smiles(test_smiles, aux=test_aux)
    if qptuna_model.transform is not None:
        expected = qptuna_model.transform.reverse_transform(expected)

    test_points = [
        Datapoint(
            smiles=smi,
            expected=round(expval.item(), ndigits=3),  # item() converts numpy to float.
            predicted=round(predval.item(), ndigits=3),
        )
        for smi, expval, predval in zip(test_smiles, expected, predicted)
    ]

    try:
        cal_points = [
            Calpoint(
                bin_edges=round(bin_edges.item(), ndigits=3),
                frac_true=round(frac_true.item(), ndigits=3),
                frac_pred=round(frac_pred.item(), ndigits=3),
            )
            for bin_edges, frac_true, frac_pred in calibration_analysis(
                expected, predicted
            )
        ]
    except ValueError:
        cal_points = ""

    data = BuildTrackingData(
        response_column_name=buildconfig.data.response_column,
        test_scores=rounded_test_scores,
        test_points=test_points,
        cal_points=cal_points,
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
        logger.warning(f"Failed to report build results {json_data} to {url}: {e}")
