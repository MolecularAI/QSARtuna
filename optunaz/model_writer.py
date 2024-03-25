import abc
import os
import time
import pickle
import types
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict, Optional, Any

import numpy
import numpy as np
import pandas as pd

from optunaz.config import ModelMode
from optunaz.utils import md5_hash
from optunaz.utils.preprocessing.transform import ModelDataTransform, AnyAuxTransformer
from optunaz.config.buildconfig import BuildConfig
from optunaz.descriptors import (
    descriptor_from_config,
    AnyDescriptor,
    SmilesBasedDescriptor,
    SmilesAndSideInfoFromFile,
    SmilesFromFile,
)
from apischema import serialize


class ModelPersistenceMode(str, Enum):
    SKLEARN_WITH_OPTUNAZ = "sklearn_with_optunaz"
    # Other options: upload to PIP? ONNX?


def predict(
    estimator,
    mode: ModelMode,
    descriptor: AnyDescriptor,
    smiles: Union[str, List[str]],
    uncert: bool = False,
    explain: bool = False,
) -> numpy.ndarray:
    """Returns prediction for a given SMILES string or a list of strings.

    For classification models,
    returns probability of class True (the second class)
    instead of just a binary value.

    Returns: 1d numpy array, with numpy.nan for invalid SMILES.

    Deprecated, use 'QSARtunaModel.predict_from_smiles()' instead.
    """

    if isinstance(smiles, str):  # Single SMILES string - wrap into list.
        smiles = [smiles]

    # Scikit-learn predicts in "batches" by taking a 2d numpy array as input. Our input can contain None if SMILES
    # are invalid, ruining the whole batch. Instead of sending a batch, we run prediction one-by-one for each molecule.
    def predict_one(fp):
        # Scikit-learn models can not handle None or NaN as inputs.
        # Treat them separately here.
        if fp is None or pd.isnull(np.array(fp)).any():
            return numpy.array([numpy.nan])  # 1d array with one element, NaN.

        # Scikit-learn wants 2d array, since 1d array is ambiguous (is it one observation with N features, or is it N
        # observations with 1 feature each). Reshape here to one row.
        fp2d = fp.reshape((1, -1))

        if uncert:
            return estimator.predict_uncert(fp2d).flatten()
        elif explain:
            from optunaz.explainability import ExplainPreds

            return ExplainPreds(estimator, fp2d, mode, descriptor)
        elif mode == ModelMode.REGRESSION:
            # Call flatten() as a workaround for 1d vs 2d output issue:
            # https://github.com/scikit-learn/scikit-learn/issues/5058
            # See also https://stackoverflow.com/a/56013213
            return estimator.predict(fp2d).flatten()
        else:
            # For classification, return probability of the second class (index 1).
            # Second class is supposed to be True or "active" or "1".
            return estimator.predict_proba(fp2d)[:, 1]

    # ChemProp models write query inputs to file and do their own batching.
    # We send all for prediction in this function for this reason.
    def predict_all(smiles_):
        if uncert:
            return estimator.predict_uncert(smiles_).flatten()
        elif explain:
            from optunaz.explainability import ExplainPreds

            return ExplainPreds(estimator, smiles_, mode, descriptor)
        elif mode == ModelMode.REGRESSION:
            return estimator.predict(smiles_).flatten()
        else:
            return estimator.predict_proba(smiles_)[:, 1]

    # ChemProp predictions/Smiles descriptors go to predict_all
    if isinstance(descriptor, SmilesBasedDescriptor) or explain:
        return predict_all(smiles)
    # All other descriptors/estimators go to predict_one
    else:
        list_of_fps = descriptor_from_config(
            smiles, descriptor, return_failed_idx=False
        )
        list_of_predictions = [predict_one(fp) for fp in list_of_fps]
        return numpy.concatenate(list_of_predictions)


def predict_from_smiles_and_descriptor(
    self, smiles: Union[str, List[str]], mode: ModelMode, uncert=False, explain=False
):
    """Returns prediction for a given SMILES string or a list of strings.

    For classification models,
    returns probability of class True (the second class)
    instead of just a binary value.

    This function will become a method of the predictive model.

    Deprecated, use 'QSARtunaModel.predict_from_smiles()' instead.
    """

    return predict(self, mode, self.mol_descriptor, smiles, uncert, explain)


def add_predict_from_smiles(model: Any, descriptor: AnyDescriptor, mode: ModelMode):
    """Adds method 'predict_from_smiles' to a predictive model.

    Deprecated, use 'QSARtunaModel.predict_from_smiles()' instead.
    """

    # side info replaced so side info not required for inference
    if isinstance(descriptor, SmilesAndSideInfoFromFile):
        descriptor = SmilesFromFile.new()
    # Add descriptor - to store in the model for convenience.
    model.mol_descriptor = descriptor

    def predict_from_smiles_method(self, smiles, mode_=mode):
        return predict_from_smiles_and_descriptor(self, smiles, mode_)

    model.predict_from_smiles = types.MethodType(predict_from_smiles_method, model)

    return model


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
class QSARtunaModel(abc.ABC):
    predictor: Predictor
    descriptor: AnyDescriptor
    mode: ModelMode
    transform: Optional[ModelDataTransform] = None
    aux_transform: Optional[AnyAuxTransformer] = None
    metadata: Optional[Dict] = None

    def predict_from_smiles(
        self,
        smiles: Union[str, List[str]],
        aux=None,
        uncert=False,
        explain=False,
        transform="default",
        aux_transform=None,
    ) -> np.ndarray:
        """Returns model predictions for the input SMILES strings.

        If some input smiles are invalid for the descriptor,
        in which case the descriptor returns None,
        those None values are not sent to the model;
        instead, NaN is used as predicted values for those invalid SMILES.
        """
        if isinstance(smiles, str):  # Single SMILES string - wrap into list.
            smiles = [smiles]

        descriptors, failed_idx = descriptor_from_config(smiles, self.descriptor)
        if aux is not None:
            aux = np.array(
                [a for a_idx, a in enumerate(aux) if a_idx not in failed_idx]
            )
            if isinstance(aux_transform, AnyAuxTransformer):
                aux = aux_transform.transform(aux)
            if len(aux.shape) == 1:
                aux = aux.reshape(len(aux), 1)
            descriptors = np.hstack((descriptors, aux))
        mask = [
            d_idx for d_idx, d in enumerate(smiles) if d_idx not in failed_idx
        ]  # Mask of valid entries.

        # First, initialize output with NaN, in case there are invalid entries.
        # Later we will overwrite valid entries with model predictions.
        n_samples = len(smiles)
        shape = (n_samples,)  # Shape of output of scikit-learn predict() functions.
        output = numpy.empty(shape)
        output.fill(np.nan)  # Default value for invalid entries.
        if len(mask) == 0:
            return output

        if uncert:
            predictions = self.predictor.predict_uncert(descriptors).flatten()
        elif explain:
            from optunaz.explainability import ExplainPreds

            return ExplainPreds(self.predictor, descriptors, self.mode, self.descriptor)
        elif self.mode == ModelMode.REGRESSION:
            # Call flatten() as a workaround for 1d vs 2d output issue:
            # https://github.com/scikit-learn/scikit-learn/issues/5058
            # See also https://stackoverflow.com/a/56013213
            predictions = self.predictor.predict(descriptors).flatten()

            # if transform is default, we use self.transform from optconfig, which can also be None
            if transform:
                if isinstance(self.transform, ModelDataTransform):
                    predictions = self.transform.reverse_transform(predictions)
                else:
                    if transform != "default":
                        raise ValueError(f"No transform function for model")
        else:
            # For classification, return probability of the second class (index 1).
            # Second class is supposed to be True or "active" or "1".
            predictions = self.predictor.predict_proba(descriptors)[:, 1]

        # Overwrite valid entries with model predictions.
        output[mask] = predictions

        return output


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
    # return modeldatatransform when applied, to reverse the transform (as long as not PTR)
    if data.log_transform and not data.probabilistic_threshold_representation:
        from optunaz.utils.preprocessing.transform import ModelDataTransform

        transform = ModelDataTransform.new(
            base=data.log_transform_base,
            negation=data.log_transform_negative,
            conversion=data.log_transform_unit_conversion,
        )
        return transform


def wrap_model(
    model: Predictor,
    descriptor: AnyDescriptor,
    mode: ModelMode,
    transform: Optional[ModelDataTransform] = None,
    aux_transform: Optional[AnyAuxTransformer] = None,
    metadata: Optional[Dict] = None,
) -> QSARtunaModel:
    return QSARtunaModel(model, descriptor, mode, transform, aux_transform, metadata)


def save_model(
    model,
    buildconfig: BuildConfig,
    filename,
    train_scores,
    test_scores,
    persist_as: ModelPersistenceMode = None,
):
    # Argument 'persist_as' became unused as of now.
    # We can deprecate/drop it,
    # or keep it in case we add other persistence modes,
    # like "upload to PIP" or "ONNX".

    descriptor = buildconfig.descriptor
    # side info replaced so side info not required for inference
    if isinstance(descriptor, SmilesAndSideInfoFromFile):
        descriptor = SmilesFromFile.new()

    mode = buildconfig.settings.mode
    metadata = get_metadata(buildconfig, train_scores, test_scores)
    try:
        transform = get_transform(buildconfig.data)
    except AttributeError:
        transform = None
    try:
        aux_transform = buildconfig.data.aux_transform
    except AttributeError:
        aux_transform = None

    model = wrap_model(model, descriptor, mode, transform, aux_transform, metadata)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(model, f)
