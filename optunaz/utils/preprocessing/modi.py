from typing import List, Tuple

import numpy as np
import sklearn.metrics
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm

from optunaz.descriptors import ECFP, descriptor_from_config
from optunaz.utils.preprocessing.splitter import stratify


def pairwise_parallel(descriptors) -> np.ndarray:
    """Calculate Tanimoto distances for descriptors matrix."""

    gen = sklearn.metrics.pairwise_distances_chunked(
        np.array(descriptors), metric="jaccard", n_jobs=-1
    )
    dists = np.concatenate(list(gen), axis=0)

    return dists


def calculate_conventional_modi_classification(labels, descriptors, distances) -> float:
    pairs_with_same_class = 0
    total_pairs = 0

    for idx in range(len(descriptors)):
        nearest_indices = np.argsort(distances[idx])[1:2]  # Exclude the point itself
        total_pairs += 1
        if labels[idx] == labels[nearest_indices[0]]:
            pairs_with_same_class += 1

    modi = pairs_with_same_class / total_pairs
    return modi


def calculate_conventional_modi_regression(labels, descriptors, distances) -> float:
    # Bin the labels
    binned_labels = stratify(labels)
    pairs_with_same_bin = 0
    total_pairs = 0

    for idx in range(len(descriptors)):
        nearest_indices = np.argsort(distances[idx])[1:2]  # Exclude the point itself
        total_pairs += 1
        if binned_labels[idx] == binned_labels[nearest_indices[0]]:
            pairs_with_same_bin += 1

    modi = pairs_with_same_bin / total_pairs
    return modi


def calculate_classification_modi_metrics(
    labels, descriptors, k: int = 3, chunk_size: int = 500
) -> Tuple[float, float, float, float]:
    """
    Calculate classification MODI (Modelability Index) metrics which are:

    modi_div (Diversity Score):

    Description: measures average pairwise distance between leave out compounds, capturing how diverse the similarities are between each compound left out vs. all other compounds

    Interpretation:
    High Score: A high diversity score (e.g., 0.8) suggests there are neighbourhoods of compounds with disagreement between activity labels.
    Low Score: A low diversity score (e.g., 0.2) indicates strong neighborhood behaviour within the data. A model will likely to perform well because similar compounds exhibit similar activity behaviour.

    modi_aci (Activity Cliff Index):

    Description: measures the average number of activity cliffs (i.e., pairs of similar compounds with different activities).

    Interpretation:
    High Score: A high activity cliff index (e.g., 0.5) indicates frequent pairs of similar compounds with different activities.
    Low Score: A low activity cliff index (e.g., 0.1) indicates low numbers of activity cliffs, indicating correlation between structre and activity spaces.

    modi_ccr (Correct Classification Rate):

    Description: measures the proportion of leave out knn predictions agreeing with the label.

    Interpretation:
    High Score: A high correct classification rate (e.g., 0.9) indicates good modelability, since leave one out structural similarity is predictive of neighboring activity.
    Low Score: A low correct classification rate (e.g., 0.5) indicates poor modelability, since leave one out structural similarity is poor at inferring neighboring activity

    conventional_modi:

    Description: measures the proportion of pairs of data points that fall into the same bin. Provides indication of how well structure space groups similar data points.

    Interpretation:
    High Score: A high conventional MODI value (e.g., 0.75) indicates good modelability, since there is agreement between structure and activity spaces.
    Low Score: A low conventional MODI value (e.g., 0.25) indicates poor modelability, since there are groups of similar structures with different activity.

    These metrics collectively provide a comprehensive view of the model's performance and the dataset's characteristics, helping you understand how well the model is likely to generalize to new data.
    """
    loo = LeaveOneOut()
    n_samples = len(labels)
    correct_predictions = np.zeros(n_samples)
    diversity_scores = np.zeros(n_samples)
    activity_cliff_counts = np.zeros(n_samples)

    descriptors = np.array(descriptors)
    distances = pairwise_parallel(descriptors)  # Compute all pairwise distances once
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(descriptors, labels)
    predictions = model.predict(descriptors)

    def process_chunk(train_indices, test_indices):
        results = []
        for train_idx, test_idx in zip(train_indices, test_indices):
            true_label = labels[test_idx]
            prediction = predictions[test_idx]
            correct_prediction = prediction == true_label
            pw_distance = distances[test_idx[0], train_idx]
            diversity_score = np.mean(pw_distance)
            nearest_indices = np.argsort(pw_distance[0])[:k]
            nearest_activities = labels[train_idx][nearest_indices]
            activity_cliff_count = np.sum(nearest_activities != true_label)
            results.append((diversity_score, activity_cliff_count, correct_prediction))
        return results

    all_train_indices, all_test_indices = zip(*loo.split(descriptors))

    # Process in chunks
    results = Parallel(n_jobs=-1)(
        delayed(process_chunk)(
            all_train_indices[i : i + chunk_size], all_test_indices[i : i + chunk_size]
        )
        for i in tqdm(range(0, n_samples, chunk_size), desc="Processing")
    )

    # Flattening the results
    results = [res for chunk in results for res in chunk]

    for i, (diversity_score, activity_cliff_count, correct_prediction) in enumerate(
        results
    ):
        diversity_scores[i] = diversity_score
        activity_cliff_counts[i] = activity_cliff_count
        correct_predictions[i] = correct_prediction

    modi_div = np.mean(diversity_scores)
    modi_aci = np.mean(activity_cliff_counts) / k
    modi_ccr = np.mean(correct_predictions)
    conventional_modi = calculate_conventional_modi_classification(
        labels, descriptors, distances
    )

    return modi_div, modi_aci, modi_ccr, conventional_modi


def calculate_regression_modi_metrics(
    labels, descriptors, k: int = 3, chunk_size: int = 50
) -> Tuple[float, float, float, float, float]:
    loo = LeaveOneOut()
    n_samples = len(labels)
    diversity_scores = np.zeros(n_samples)
    leave_out_activities = np.zeros(n_samples)
    nearest_activities = np.zeros(n_samples)

    descriptors = np.array(descriptors)
    distances = pairwise_parallel(descriptors)  # Compute all pairwise distances once
    model = KNeighborsRegressor(n_neighbors=k, n_jobs=-1).fit(descriptors, labels)
    predictions = model.predict(descriptors)

    def process_chunk(train_indices, test_indices):
        results = []
        for train_idx, test_idx in zip(train_indices, test_indices):
            prediction = predictions[test_idx]
            pw_distances = distances[test_idx[0], train_idx]
            diversity_score = np.mean(pw_distances)
            nearest_idx = np.argmin(pw_distances)
            nearest_activity = labels[nearest_idx]
            leave_out_activity = labels[test_idx]
            results.append(
                (prediction, diversity_score, nearest_activity, leave_out_activity)
            )
        return results

    all_train_indices, all_test_indices = zip(*loo.split(descriptors))

    # Process in chunks
    results = Parallel(n_jobs=-1)(
        delayed(process_chunk)(
            all_train_indices[i : i + chunk_size], all_test_indices[i : i + chunk_size]
        )
        for i in tqdm(range(0, n_samples, chunk_size), desc="Processing")
    )

    # Flattening the results
    results = [res for chunk in results for res in chunk]

    for i, (
        prediction,
        diversity_score,
        nearest_activity,
        leave_out_activity,
    ) in enumerate(results):
        predictions[i] = prediction
        diversity_scores[i] = diversity_score
        nearest_activities[i] = nearest_activity
        leave_out_activities[i] = leave_out_activity

    mse = mean_squared_error(labels, predictions)
    modi_aciR2 = r2_score(leave_out_activities, nearest_activities)
    total_variance = np.var(labels)
    modi_q2 = 1 - mse / total_variance
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    modi_ssR2 = 1 - ss_res / ss_tot
    modi_ds = np.mean(diversity_scores)
    conventional_modi = calculate_conventional_modi_regression(
        labels, descriptors, distances
    )

    return modi_ds, modi_aciR2, modi_q2, modi_ssR2, conventional_modi


def assess_modelability(
    train_smiles: List[str],
    train_y: np.ndarray,
    train_aux: np.ndarray,
    test_smiles: List[str],
    test_y: np.ndarray,
    test_aux: np.ndarray,
    response_type: str,
):
    smiles = np.concatenate((np.array(train_smiles), np.array(test_smiles)), axis=0)
    labels = np.concatenate((train_y, test_y), axis=0)
    descriptors = descriptor_from_config(
        smiles, ECFP.new(returnRdkit=False), return_failed_idx=False
    )
    if train_aux is not None and test_aux is not None:
        aux_data = np.concatenate((train_aux, test_aux), axis=0)
        descriptors = np.concatenate((descriptors, aux_data), axis=1)

    if response_type == "classification":
        (
            modi_div,
            modi_aci,
            modi_ccr,
            conventional_modi,
        ) = calculate_classification_modi_metrics(labels, descriptors)
        return {
            "modi_div": modi_div,
            "modi_aci": modi_aci,
            "modi_ccr": modi_ccr,
            "conventional_modi": conventional_modi,
        }
    elif response_type == "regression":
        (
            modi_div,
            modi_aciR2,
            modi_q2,
            modi_ssR2,
            conventional_modi,
        ) = calculate_regression_modi_metrics(labels, descriptors)
        return {
            "modi_div": modi_div,
            "modi_aciR2": modi_aciR2,
            "modi_q2": modi_q2,
            "modi_ssR2": modi_ssR2,
            "conventional_modi": conventional_modi,
        }
    else:
        return
