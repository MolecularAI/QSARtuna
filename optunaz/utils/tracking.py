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
from optunaz.model_writer import QptunaModel
from math import isnan

logger = logging.getLogger(__name__)


def get_authorization_header():
    return os.getenv("QPTUNA_TOKEN")


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


def removeprefix(line: str, prefix: str) -> str | None:
    # Starting from Python 3.9, str has method removeprefix().
    # We target Python 3.7+, so here is this function.
    if line.startswith(prefix):
        return line[len(prefix) :]


def round_scores(test_scores):
    return {
        k: [round(v, ndigits=3) if not isnan(v) else None for v in vs]
        for k, vs in test_scores.items()
    }


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
                logger.warning(
                    f"Failed to report Optimization results {json_data} to {url}: {e}"
                )
        except Exception as e:
            logger.warning(f"Failed to calculate Optimization results: {e}")


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
class Uqpoint:
    smiles: str
    error: float
    unc: float


@dataclass
class BuildTrackingData:
    """Dataclass defining internal Build tracking format"""

    response_column_name: str
    train_scores: Dict[str, float] | None
    test_scores: Dict[str, float] | None
    train_points: List[Datapoint] | None
    test_points: List[Datapoint] | None
    cal_points: List[Calpoint] | None
    test_uq_points: List[Uqpoint] | None
    train_uq_points: List[Uqpoint] | None
    train_uq_cal_points: List[Calpoint] | None
    test_uq_cal_points: List[Calpoint] | None


def track_build(qptuna_model: QptunaModel):
    predictor = qptuna_model.predictor
    train_smiles, test_smiles = predictor.train_smiles_, predictor.test_smiles_
    train_expected, test_expected = predictor.y_, predictor.test_y_
    train_predicted, test_predicted = predictor.train_pred, predictor.test_pred
    train_error, test_error = predictor.train_err, predictor.test_err
    train_unc, test_unc = predictor.train_unc, predictor.test_unc
    train_scores = qptuna_model.metadata.get("train_scores", None)
    test_scores = qptuna_model.metadata.get("test_scores", None)
    buildconfig = qptuna_model.metadata.get("buildconfig")

    if qptuna_model.transform is not None:
        train_predicted = qptuna_model.transform.reverse_transform(train_predicted)
        test_predicted = qptuna_model.transform.reverse_transform(test_predicted)
        test_expected = qptuna_model.transform.reverse_transform(test_expected)
        train_expected = qptuna_model.transform.reverse_transform(train_expected)

    rounded_train_scores = (
        {k: round(v, 3) for k, v in train_scores.items()}
        if train_scores is not None
        else None
    )

    train_points = [
        Datapoint(smi, round(expval.item(), 3), round(predval.item(), 3))
        for smi, expval, predval in zip(train_smiles, train_expected, train_predicted)
    ]

    if test_smiles is not None and len(test_smiles) >= 1:
        rounded_test_scores = (
            {k: round(v, 3) for k, v in test_scores.items()}
            if test_scores is not None
            else None
        )

        test_points = [
            Datapoint(smi, round(expval.item(), 3), round(predval.item(), 3))
            for smi, expval, predval in zip(test_smiles, test_expected, test_predicted)
        ]

    else:
        rounded_test_scores = None
        test_points = None

    try:
        cal_points = [
            Calpoint(
                round(bin_edges.item(), 3),
                round(frac_true.item(), 3),
                round(frac_pred.item(), 3),
            )
            for bin_edges, frac_true, frac_pred in calibration_analysis(
                test_expected, test_predicted
            )
        ]
    except ValueError:
        cal_points = None

    if test_unc is not None:
        test_uq_points = [
            Uqpoint(smi, round(error.item(), 3), round(unc.item(), 3))
            for smi, error, unc in zip(test_smiles, test_error, test_unc)
        ]
        try:
            test_uq_cal_points = [
                Calpoint(
                    round(bin_edges.item(), 3),
                    round(frac_true.item(), 3),
                    round(frac_pred.item(), 3),
                )
                for bin_edges, frac_true, frac_pred in calibration_analysis(
                    test_error, test_unc, norm=True
                )
            ]
        except ValueError:
            test_uq_cal_points = None
    else:
        test_uq_points, test_uq_cal_points = None, None

    if train_unc is not None:
        train_uq_points = [
            Uqpoint(smi, round(error.item(), 3), round(unc.item(), 3))
            for smi, error, unc in zip(train_smiles, train_error, train_unc)
        ]
        try:
            train_uq_cal_points = [
                Calpoint(
                    round(bin_edges.item(), 3),
                    round(frac_true.item(), 3),
                    round(frac_pred.item(), 3),
                )
                for bin_edges, frac_true, frac_pred in calibration_analysis(
                    train_unc, train_error, norm=True
                )
            ]
        except ValueError:
            train_uq_cal_points = None
    else:
        train_uq_points, train_uq_cal_points = None, None

    data = BuildTrackingData(
        response_column_name=buildconfig["data"]["response_column"],
        test_scores=rounded_test_scores,
        train_scores=rounded_train_scores,
        train_points=train_points,
        test_points=test_points,
        cal_points=cal_points,
        test_uq_points=test_uq_points,
        train_uq_points=train_uq_points,
        train_uq_cal_points=train_uq_cal_points,
        test_uq_cal_points=test_uq_cal_points,
    )

    json_data = serialize(data)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": get_authorization_header(),
    }
    url = buildconfig["settings"]["tracking_rest_endpoint"]

    try:
        response = requests.post(url, json=json_data, headers=headers)
        logger.info(f"Reported build results {json_data} to {url}")
    except Exception as e:
        logger.warning(f"Failed to report build results {json_data} to {url}: {e}")
