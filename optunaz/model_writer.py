import abc
import os
import dill
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from apischema import serialize

from optunaz.algorithms.chem_prop import BaseChemProp
from optunaz.algorithms.fast_prop import BaseFastProp
from optunaz.config import ModelMode
from optunaz.config.buildconfig import BuildConfig
from optunaz.descriptors import (
    AnyDescriptor,
    SmilesAndSideInfoFromFile,
    SmilesFromFile,
    combine_covariates,
    descriptor_from_config,
)
from optunaz.utils.active_learning.similarity_to_train import compute_nearest_neighbor_similarity

from optunaz.utils import md5_hash
from optunaz.utils.preprocessing.transform import (
    AnyAuxTransformer,
    ModelDataTransform,
    PTRTransform,
)


class Predictor(abc.ABC):
    """Interface definition for scikit-learn/chemprop Predictor.

    Scikit-learn does not define a class that describes the Predictor interface.
    Instead, scikit-learn describes in text that Predictor should have method
    'predict', and optionally 'predict_proba':
    https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects

    This class describes this interface as an abstract Python class,
    for convenience and better type checking.
    """

    @abc.abstractmethod
    def predict(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Returns predicted values."""
        ...

    def predict_proba(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """For Classification algorithms, returns algorithmic posterior of a prediction.

        This method is optional, and is not marked with @abstractmethod.
        """
        ...

    def predict_uncert(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """For supported algorithms, quantifies uncertainty of a prediction.

        This method is optional, and is not marked with @abstractmethod.
        """
        ...

    def explain(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Explains a prediction.

        This method is optional, and is not marked with @abstractmethod.
        """
        ...


@dataclass
class QptunaModel(abc.ABC):
    predictor: Predictor
    descriptor: AnyDescriptor
    mode: ModelMode
    transform: Optional[ModelDataTransform] = None
    aux_transform: Optional[AnyAuxTransformer] = None
    metadata: Optional[Dict] = None

    def predict_from_smiles(
        self,
        smiles: Union[str, List[str]],
        aux: np.ndarray | None = None,
        uncert: bool = False,
        sim_to_train: bool = False,
        similarity_metric: str = "ecfp_tanimoto",
        explain: bool = False,
        transform: ModelDataTransform | str = "default",
        aux_transform: AnyAuxTransformer | None = None,
    ) -> np.ndarray | List:
        """Returns model predictions for the input SMILES strings.

        If some input smiles are invalid for the descriptor,
        in which case the descriptor returns None,
        those None values are not sent to the model;
        instead, NaN is used as predicted values for those invalid SMILES.
        """

        if isinstance(smiles, str):  # Single SMILES string - wrap into list.
            smiles = [smiles]

        if self.metadata is not None:
            self.set_version_specific_params()

        descriptors, failed_idx = descriptor_from_config(smiles, self.descriptor)

        mask = np.array(
            [d_idx for d_idx, d in enumerate(smiles) if d_idx not in failed_idx]
        )  # Mask of valid entries.
        if aux is not None:
            aux = np.array(aux)[mask]
            if isinstance(aux_transform, AnyAuxTransformer):
                aux = aux_transform.transform(aux)
            descriptors = combine_covariates(self.descriptor, descriptors, aux)

        # First, initialize output with NaN, in case there are invalid entries.
        # Later we will overwrite valid entries with model predictions.
        n_samples = len(smiles)
        shape = (n_samples,)  # Shape of output of scikit-learn predict() functions.
        output = np.empty(shape)
        output.fill(np.nan)  # Default value for invalid entries.
        if len(mask) == 0:
            return output

        if explain:
            from optunaz.explainability import ExplainPreds

            return ExplainPreds(self.predictor, descriptors, self.mode, self.descriptor)
        
        if uncert:
            unc_output = output.copy()
            sim_output_sim = output.copy()
            sim_output_pred = output.copy()
            sim_output_nn = np.empty(shape, dtype="<U1000")

            has_latent_space = isinstance(self.predictor, (BaseFastProp, BaseChemProp))
            if has_latent_space:
                predictions, unc_predictions = self.predictor.predict_uncert(
                    descriptors
                )
            else:
                unc_predictions = self.predictor.predict_uncert(descriptors)

            if sim_to_train:
                smiles_masked = [smiles[mask_idx] for mask_idx in mask]

                (
                    nearest_neighbor_smiles,
                    nearest_neighbor_similarity,
                    nearest_neighbor_prediction,
                ) = compute_nearest_neighbor_similarity(
                    self, smiles_masked, descriptors, similarity_metric, has_latent_space
                )

        if self.mode == ModelMode.REGRESSION:
            # Call flatten() as a workaround for 1d vs 2d output issue:
            # https://github.com/scikit-learn/scikit-learn/issues/5058
            # See also https://stackoverflow.com/a/56013213
            if not isinstance(self.predictor, (BaseFastProp, BaseChemProp)) or not uncert:
                predictions = self.predictor.predict(descriptors).flatten()

            # If transform is "default", we use self.transform from optconfig, which is allowed to be None
            # Setting transform to False would avoid transforming
            if transform:
                if isinstance(self.transform, ModelDataTransform):
                    # Allow PTR to reverse transform, if set
                    predictions = perform_ptr(
                        self.metadata, self.transform, predictions
                    )
                    # Log transform can then be performed
                    predictions = self.transform.reverse_transform(predictions)
                    if uncert:
                        unc_predictions = perform_ptr(
                            self.metadata, self.transform, unc_predictions
                        )
                        unc_predictions = self.transform.reverse_transform(
                            unc_predictions
                        )
                else:
                    if transform != "default":
                        raise ValueError(f"No transform function for model")
        else:  # ModelMode.CLASSIFICATION
            # For classification, return probability of the second class (index 1).
            # Second class is supposed to be True or "active" or "1".
            if not isinstance(self.predictor, (BaseFastProp, BaseChemProp)) or not uncert:
                predictions = self.predictor.predict_proba(descriptors)[:, 1]

        # Overwrite valid entries with model predictions.
        output[mask] = predictions

        if uncert:
            unc_output[mask] = unc_predictions.flatten()
            if sim_to_train:
                sim_output_sim[mask] = nearest_neighbor_similarity.flatten()
                sim_output_nn[mask] = nearest_neighbor_smiles.flatten()
                sim_output_pred[mask] = nearest_neighbor_prediction.flatten()
                return (
                    output,
                    unc_output,
                    sim_output_sim,
                    sim_output_nn,
                    sim_output_pred,
                )
            else:
                return output, unc_output
        else:
            return output

    def set_version_specific_params(self):
        pass



def is_version_less_than(version, target_version):
    version_parts = list(map(int, version.split(".")))
    target_parts = list(map(int, target_version.split(".")))
    return version_parts < target_parts


def get_metadata(buildconfig, train_scores, test_scores):
    """Metadata for a predictive model."""
    from optunaz import __version__

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        name = buildconfig.metadata.name
    except AttributeError:
        name = ""

    metadata = {
        "name": name,
        "buildconfig": serialize(buildconfig),
        "version": __version__,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "date_created": timestamp,
    }
    metadata["hash"] = md5_hash(metadata)
    return metadata


def get_transform(data):
    # return modeldatatransform when applied for a reverse transform
    if data.log_transform:
        transform = ModelDataTransform.new(
            base=data.log_transform_base,
            negation=data.log_transform_negative,
            conversion=data.log_transform_unit_conversion,
        )
        return transform


def perform_ptr(metadata, transform, predictions):
    # return PTR transform when applied for a reverse transform
    build_data = serialize(metadata)["buildconfig"]["data"]
    if build_data["probabilistic_threshold_representation"]:
        threshold = build_data["probabilistic_threshold_representation_threshold"]
        std = build_data["probabilistic_threshold_representation_std"]
        if isinstance(transform, ModelDataTransform):
            threshold = transform.transform(threshold)
            std = transform.transform(std)

        ptr_transform = PTRTransform.new(
            threshold=threshold,
            std=std,
        )
        return ptr_transform.reverse_transform(predictions)
    else:
        return predictions


def wrap_model(
    model: Predictor,
    descriptor: AnyDescriptor,
    mode: ModelMode,
    transform: Optional[ModelDataTransform] = None,
    aux_transform: Optional[AnyAuxTransformer] = None,
    metadata: Optional[Dict] = None,
) -> QptunaModel:
    return QptunaModel(model, descriptor, mode, transform, aux_transform, metadata)


def save_model(
    model,
    buildconfig: BuildConfig,
    filename,
    train_scores,
    test_scores,
):
    descriptor = buildconfig.descriptor
    # side info replaced where possible as not required for inference
    # models trained with auxiliary molecule descriptors require new inference files
    if (
        isinstance(descriptor, SmilesAndSideInfoFromFile)
        and descriptor.x_aux is None
        and buildconfig.data.covariate_column == None
    ):
        descriptor = SmilesFromFile.new()

    metadata = get_metadata(buildconfig, train_scores, test_scores)
    transform = (
        get_transform(buildconfig.data) if hasattr(buildconfig, "data") else None
    )
    aux_transform = (
        getattr(buildconfig.data, "aux_transform", None)
        if hasattr(buildconfig, "data")
        else None
    )

    model = wrap_model(
        model, descriptor, buildconfig.settings.mode, transform, aux_transform, metadata
    )

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(filename, "wb") as f:
        dill.dump(model, f)

    return model
