from __future__ import annotations

import logging
import numpy as np

from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

from optunaz.descriptors import (
    ECFP,
    descriptor_from_config,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple
    from optunaz.model_writer import QptunaModel


def compute_nearest_neighbor_similarity(
    model: QptunaModel,
    smiles: List[str],
    descriptors: np.ndarray,
    similarity_metric: str,
    has_latent_space: bool,
) -> Tuple[np.ndarray, ...]:
    """
    Compute the nearest neighbor similarity as specified by the input parameters.

    Args:
        model: Qptuna model with predictor and descriptors.
        smiles: List of SMILES
        descriptors: Model descriptors of each SMILES
        similarity_metric: Which metric to use for computing similarity (either 'ecfp_tanimoto', or
            'model_descriptor')
        has_latent_space: Whether the model has latent space or not.
    Returns:
        The nearest neighbor SMILES, nearest neighbor similarity, and nearest neighbor prediction.
    """
    if similarity_metric not in ["model_descriptor", "ecfp_tanimoto"]:
        raise ValueError(
            f"similarity_metric must be one of ['ecfp_tanimoto', 'model_descriptor'] not {similarity_metric}"
        )

    if similarity_metric == "model_descriptor" and has_latent_space:
        return latent_space_similarity(model, descriptors)
    elif similarity_metric == "model_descriptor":
        return descriptor_space_similarity(model, descriptors)

    return ecfp_tanimoto_similarity(model, smiles)


def descriptor_space_similarity(
    model: QptunaModel, inference_descriptors: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Calculate the nearest similarity between train and inference sets in the model descriptor space.

    Args:
        model: Qptuna model with predictor and descriptors.
        inference_descriptors: Descriptors of the inference/test data set.
    Returns:
        The nearest neighbor SMILES, nearest neighbor similarity, and nearest neighbor prediction.
    """
    validate = _validate_sim_to_train(model, inference_descriptors)
    if validate:
        return validate

    if isinstance(inference_descriptors, bool):
        sim_mat = _fast_jaccard_score(inference_descriptors, model.predictor.X_)
        axis = 1
    else:
        sim_mat = cosine_similarity(model.predictor.X_, inference_descriptors)
        axis = 0

    nearest_neighbor_smiles = model.predictor.train_smiles_[
        np.argmax(sim_mat, axis=axis)
    ]
    nearest_neighbor_similarity = sim_mat.max(axis=axis)
    nearest_neighbor_prediction = model.predict_from_smiles(nearest_neighbor_smiles)
    return (
        nearest_neighbor_smiles,
        nearest_neighbor_similarity,
        nearest_neighbor_prediction,
    )


def ecfp_tanimoto_similarity(
    model: QptunaModel, input_smiles: List[str]
) -> Tuple[np.ndarray, ...]:
    """
    Calculate the nearest similarity between train & test sets using ECFP4 fingerprints and
    Tanimoto similarity.

    Args:
        model: Qptuna model with predictor and descriptors.
        input_smiles: List of smiles in the test set.
    Returns:
        The nearest neighbor SMILES, nearest neighbor similarity, and nearest neighbor prediction.
    """
    n_input_samples = len(input_smiles)
    nearest_neighbor_smiles = np.zeros(n_input_samples, dtype="object")
    nearest_neighbor_similarity = np.nan * np.ones(n_input_samples)
    nearest_neighbor_prediction = np.nan * np.ones(n_input_samples)

    validate = _validate_sim_to_train(model, input_smiles)
    if validate:
        return validate

    ecfp = ECFP.new(radius=2, nBits=1024)

    input_features, failed_idx = descriptor_from_config(
        input_smiles, ecfp, return_failed_idx=True
    )
    mask = np.array(
        [idx for idx in np.arange(n_input_samples, dtype=int) if idx not in failed_idx]
    )  # Mask of valid entries.

    train_smiles = _get_unique_valid_smiles(model.predictor.train_smiles_)
    train_set_features, failed_train_idx = descriptor_from_config(
        train_smiles,
        ecfp,
        return_failed_idx=True,
    )
    mask_train = np.array(
        [
            idx
            for idx in np.arange(len(train_smiles), dtype=int)
            if idx not in failed_train_idx
        ]
    )  # Mask of valid entries.

    similarity_matrix = _fast_jaccard_score(input_features, train_set_features)
    axis = 1

    nearest_neighbor_smiles[mask] = train_smiles[mask_train][
        np.argmax(similarity_matrix, axis=axis)
    ]
    nearest_neighbor_similarity[mask] = similarity_matrix.max(axis=axis)
    nearest_neighbor_prediction[mask] = model.predict_from_smiles(
        nearest_neighbor_smiles[mask]
    )

    return (
        nearest_neighbor_smiles,
        nearest_neighbor_similarity,
        nearest_neighbor_prediction,
    )


def latent_space_similarity(
    model: QptunaModel, inference_descriptors: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """
    Calculate the nearest latent similarity between train & inference

    Args:
        model: Qptuna model with predictor and descriptors.
        inference_descriptors: Descriptors of the inference/test data set.
    Returns:
        The nearest neighbor SMILES, nearest neighbor similarity, and nearest neighbor prediction.
    """
    train_fps = model.predictor.chemprop_fingerprint(
        model.predictor.train_smiles_, fingerprint_type="last_FFN"
    )
    inf_fps = model.predictor.chemprop_fingerprint(
        inference_descriptors, fingerprint_type="last_FFN"
    )
    sim_mat = cosine_similarity(train_fps, inf_fps)
    nearest_neighbor_smiles = model.predictor.train_smiles_[np.argmax(sim_mat, axis=0)]
    nearest_neighbor_similarity = sim_mat.max(axis=0)
    nearest_neighbor_prediction = model.predict_from_smiles(nearest_neighbor_smiles)
    return (
        nearest_neighbor_smiles,
        nearest_neighbor_similarity,
        nearest_neighbor_prediction,
    )


def _fast_jaccard_score(
    feature_set1: np.ndarray, feature_set2: np.ndarray
) -> np.ndarray:
    """Compute the pairwise jaccard score between two sets of features."""
    n_cores = effective_n_jobs(-1)
    similarity_calculator = lambda inp: [
        jaccard_score(tr, inp, average="macro") for tr in feature_set2
    ]
    similarity_matrix = np.array(
        Parallel(n_jobs=n_cores, prefer="threads")(
            delayed(similarity_calculator)(feature_set1[i])
            for i in range(len(feature_set1))
        )
    )
    return similarity_matrix


def _validate_sim_to_train(
    model: QptunaModel, inference_descriptors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate custom models that may not have X or train_smiles in predictor attributes"""

    if not hasattr(model.predictor, "X_") or not hasattr(
        model.predictor, "train_smiles_"
    ):
        logging.warning(
            "sim-to-train requires predictor to have 'X_' and 'train_smiles_' attributes"
        )
        ret = np.full(inference_descriptors.shape[0], np.nan)
        return np.empty(inference_descriptors.shape[0], dtype="<U1000"), ret


def _get_unique_valid_smiles(smiles_list: np.ndarray) -> np.ndarray:
    smiles_list = np.unique(smiles_list)

    smiles_list = np.array(
        [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
    )
    return smiles_list
