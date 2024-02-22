import logging
import numpy as np
import sklearn.model_selection


from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)


def validate_cls_input(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Validate true and predicted arrays for metrics.
    """
    assert len(y_true) == len(y_pred), "Class labels and predictions do not match"
    assert np.array_equal(
        np.unique(y_true).astype(int), [0, 1]
    ), f"Class labels must be binary: {np.unique(y_true)}"

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    yt = yt.flatten()
    yp = yp.flatten()

    return yt, yp


def auc_pr_cal(y_true: np.ndarray, y_pred: np.ndarray, pi_zero: float = 0.1) -> float:
    """Compute calibrated AUC PR metric.

    Implemented according to MELLODDY SparseChem https://github.com/melloddy/SparseChem.
    Calibration modifies the AUC PR to account for class imbalance.
    """

    try:
        yt, yp = validate_cls_input(y_true, y_pred)
    except AssertionError:
        return None

    num_pos_va = yt.sum()
    num_neg_va = (yt == 0).sum()
    pos_rate = num_pos_va / (num_pos_va + num_neg_va)
    pos_rate_ref = pi_zero
    pos_rate = np.clip(pos_rate, 0, 0.99)
    cal_fact_aucpr = pos_rate * (1 - pos_rate_ref) / (pos_rate_ref * (1 - pos_rate))

    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(
        y_true=yt, probas_pred=yp
    )
    with np.errstate(divide="ignore"):
        # precision can be zero but can be ignored so disable warnings (divide by 0)
        precision_cal = 1 / (((1 / precision - 1) * cal_fact_aucpr) + 1)

    return sklearn.metrics.auc(x=recall, y=precision_cal)


def bedroc_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0) -> float:
    """Compute BEDROC metric.

    Implemented according to Truchon, J. & Bayly, C.I. Evaluating Virtual Screening Methods:
    Good and Bad Metric for the “Early Recognition” Problem.
    J. Chem. Inf. Model. 47, 488-508 (2007).
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    try:
        yt, yp = validate_cls_input(y_true, y_pred)
    except AssertionError:
        return None

    scores = list(zip(yt, yp))
    scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

    return CalcBEDROC(scores, 0, alpha)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Concordance index.

    Statistical metric to indicate the quality of a predicted ranking based on Harald,
    et al. "On ranking in survival analysis: Bounds on the concordance index."
    Advances in neural information processing systems (2008): 1209-1216.
    """

    try:
        yt, yp = validate_cls_input(y_true, y_pred)
    except AssertionError:
        return None

    idx = np.argsort(yt)
    yt = yt[idx]
    yp = yp[idx]

    pairs = 0
    correct_pairs = 0.0

    for i in range(len(yt)):
        true_a = yt[i]
        pred_a = yp[i]

        for j in range(i + 1, len(yt)):
            true_b = yt[j]
            pred_b = yp[j]
            if true_a != true_b:
                pairs += 1
                if pred_a == pred_b:
                    correct_pairs += 0.5
                elif pred_a < pred_b:
                    correct_pairs += true_a < true_b
                else:
                    correct_pairs += true_a > true_b

    assert pairs > 0, "Insufficient indexes for comparison"

    return correct_pairs / pairs
