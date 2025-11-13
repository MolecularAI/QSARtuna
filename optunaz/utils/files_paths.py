import os
from typing import List, Optional

from joblib import Memory

from optunaz.datareader import Dataset
from optunaz.descriptors import (
    CompositeDescriptor,
    MolDescriptor,
    ScaledDescriptor,
    UnfittedSklearnScaler,
)


def move_up_directory(path, n=1):
    """Function, to move up "n" directories for a given "path"."""
    # add +1 to take file into account
    if os.path.isfile(path):
        n += 1
    for _ in range(n):
        path = os.path.dirname(os.path.abspath(path))
    return path


def attach_root_path(path):
    """Function to attach the root path of the module for a given "path"."""
    ROOT_DIR = move_up_directory(os.path.abspath(__file__), n=2)
    return os.path.join(ROOT_DIR, path)


def copy_path_for_scaled_descriptor(
    descriptor_input: List[MolDescriptor] | MolDescriptor,
    dataset: Dataset,
    cache: Optional[Memory] = None,
) -> None:
    """Ensures that the scaler data for the provided molecular descriptors is properly set up.
    If the descriptor is a `ScaledDescriptor` with an unfitted scaler, it initializes the scaler
    with the dataset's training file and input column. If the scaler is already fitted, it ensures
    the scaler is ready for use. For composite descriptors, the function recursively processes
    all contained descriptors.
    """

    def process_descriptor(descriptor: MolDescriptor) -> None:
        if isinstance(descriptor, ScaledDescriptor):
            scaler = descriptor.parameters.scaler
            if (
                isinstance(scaler, UnfittedSklearnScaler)
                and not scaler.mol_data.file_path
            ):
                descriptor.set_unfitted_scaler_data(
                    dataset.training_dataset_file, dataset.input_column, cache=cache
                )
            else:
                descriptor._ensure_scaler_is_fitted(cache=cache)
        elif isinstance(descriptor, CompositeDescriptor):
            for sub_descriptor in descriptor.parameters.descriptors:
                process_descriptor(sub_descriptor)

    descriptors = (
        descriptor_input if isinstance(descriptor_input, list) else [descriptor_input]
    )
    for descriptor in descriptors:
        process_descriptor(descriptor)
