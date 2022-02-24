import functools
import os
import pickle
import types
from enum import Enum
from typing import Union, List, Any

import dill
import numpy

from optunaz.config import ModelMode
from optunaz.descriptors import MolDescriptor


class ModelPersistenceMode(str, Enum):
    PLAIN_SKLEARN = "plain_sklearn"
    SKLEARN_WITH_OPTUNAZ = "sklearn_with_optunaz"
    # Third option: container compatible with AZ PIP.


def predict(
    estimator, mode: ModelMode, descriptor: MolDescriptor, smiles: Union[str, List[str]]
) -> numpy.ndarray:
    """Returns prediction for a given SMILES string or a list of strings.

    For classification models,
    returns probability of class True (the second class)
    instead of just a binary value.

    Returns: 1d numpy array.
    """
    if isinstance(smiles, str):  # Single SMILES string - wrap into list.
        smiles = [smiles]

    list_of_arrays = [descriptor.calculate_from_smi(smi) for smi in smiles]

    array2d = numpy.stack(list_of_arrays, axis=0)

    if mode == ModelMode.REGRESSION:
        # Call flatten() as a workaround for
        # https://github.com/scikit-learn/scikit-learn/issues/5058
        # See also https://stackoverflow.com/a/56013213
        result = estimator.predict(array2d).flatten()
    else:
        # For classification, return probability of the second class (index 1).
        # Second class is supposed to be True or "active" or "1".
        result = estimator.predict_proba(array2d)[:, 1]

    return result


def predict_from_smiles_and_descriptor(
    self, smiles: Union[str, List[str]], mode: ModelMode
):
    """Returns prediction for a given SMILES string or a list of strings.

    For classification models,
    returns probability of class True (the second class)
    instead of just a binary value.

    This function will become a method of the predictive model.
    """

    return predict(self, mode, self.mol_descriptor, smiles)


def add_predict_from_smiles(model: Any, descriptor: MolDescriptor, mode: ModelMode):
    """Adds method 'predict_from_smiles' to a predictive model."""

    # Add descriptor - to store in the model for convenience.
    model.mol_descriptor = descriptor

    # Add method to an object: https://stackoverflow.com/a/2982
    method = functools.partial(predict_from_smiles_and_descriptor, mode=mode)
    model.predict_from_smiles = types.MethodType(method, model)

    return model


def save_model(
    model,
    descriptor,
    mode: ModelMode,
    filename,
    persist_as: ModelPersistenceMode = None,
):
    if persist_as is None:
        persist_as = ModelPersistenceMode.SKLEARN_WITH_OPTUNAZ

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if persist_as == ModelPersistenceMode.PLAIN_SKLEARN:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
    elif persist_as == ModelPersistenceMode.SKLEARN_WITH_OPTUNAZ:
        model = add_predict_from_smiles(model, descriptor, mode)
        with open(filename, "wb") as f:
            dill.dump(model, f, recurse=True, byref=False)
    else:
        raise ValueError(f"Unrecognized mode: {persist_as}")
