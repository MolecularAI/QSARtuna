import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Iterable, Optional, Union, Any

import numpy as np
import pandas as pd
from apischema import schema
from apischema.metadata import skip
from rdkit import Chem

from optunaz.utils.preprocessing.deduplicator import (
    Deduplicator,
    KeepAllNoDeduplication,
)
from optunaz.utils.preprocessing.splitter import (
    Stratified,
    NoSplitting,
    Splitter,
)

logger = logging.getLogger(__name__)


def isvalid(smiles: Iterable[str]) -> np.ndarray:
    valid = [Chem.MolFromSmiles(smi) is not None for smi in smiles]
    return np.array(valid, dtype=bool)


def read_data(
    filename: Union[Path, str],
    smiles_col: str,
    resp_col: str = None,
    shuffle: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Reads data, drops NaNs and invalid SMILES.

    Returns a tuple of ( SMILES (X), responses (Y) ).
    """

    if filename is None:
        return [], np.empty((0,))

    df = pd.read_csv(filename, skipinitialspace=True)

    # remove rows, that hold "NaN"s in the respective columns of interest
    nrow_before = df.shape[0]
    if resp_col is not None:
        df = df[[smiles_col, resp_col]].dropna()
    else:
        df = df[[smiles_col]].dropna()
    if df.shape[0] < nrow_before:
        logger.info(f"Removed {nrow_before - df.shape[0]} rows with NaNs.")

    # if specified, shuffle the data (the rows) to make sure the X-validation is done on a proper distribution
    # TODO(alex): move to cross-validation.
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Drop invalid. For 'index' trick, see https://stackoverflow.com/a/27360130.
    smiles = df[smiles_col]
    dropmask = ~isvalid(smiles)
    df.drop(df[dropmask].index, inplace=True)

    valid_smiles = df[smiles_col].to_list()
    valid_responses = df[resp_col].to_numpy() if resp_col else np.empty((0,))

    return valid_smiles, valid_responses


def deduplicate(
    train_smiles, train_y, deduplication_strategy: Deduplicator
) -> Tuple[List[str], np.ndarray]:
    """Remove duplicates based on RDKit canonical SMILES representation."""

    smicol = "SMILES"
    cancol = "canonical_SMILES"
    respcol = "response"

    can_smiles = [
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False, canonical=True)
        for smi in train_smiles
    ]

    df = pd.DataFrame({smicol: train_smiles, cancol: can_smiles, respcol: train_y})

    df = deduplication_strategy.dedup(df, cancol)

    return df[smicol], df[respcol]


def split(
    train_smiles, train_y, strategy: Splitter
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    smicol = "SMILES"
    respcol = "response"
    df = pd.DataFrame({smicol: train_smiles, respcol: train_y})
    if hasattr(strategy, "respcol"):  # Stratified has respcol.
        strategy.respcol = respcol
    train, test = strategy.split(df)
    return (
        train[smicol].to_list(),
        train[respcol].to_numpy(),
        test[smicol].to_list(),
        test[respcol].to_numpy(),
    )


def merge(
    train_smiles: List[str],
    train_y: np.ndarray,
    test_smiles: List[str],
    test_y: np.ndarray,
) -> Tuple[List[str], np.ndarray]:

    merged_smiles = train_smiles + test_smiles
    merged_y = np.concatenate((train_y, test_y), axis=None)

    return merged_smiles, merged_y


@dataclass
class Dataset:
    """Dataset.

    Holds training data, optional test data, and names of input and response columns.
    """

    training_dataset_file: str = field(
        metadata=schema(
            title="Main dataset file",
            description="Main dataset file. This is training dataset if no splitting stratetgy is selected. If splitting strategy is selected, this dataset is split into training dataset and test dataset.",
        )
    )
    input_column: str = field(
        metadata=schema(
            title="Input column",
            description="Name of input column with SMILES strings.",
        )
    )
    response_column: str = field(
        metadata=schema(
            title="Response column",
            description="Name of response column with the value that the model will try to predict.",
        )
    )
    deduplication_strategy: Deduplicator = field(
        default_factory=KeepAllNoDeduplication,
        metadata=schema(title="Deduplication strategy"),
    )
    split_strategy: Splitter = field(
        default_factory=NoSplitting, metadata=schema(title="Data splitting strategy")
    )
    test_dataset_file: Optional[str] = field(
        default=None,
        metadata=schema(
            title="Additional test dataset file",
            description="Additional holdout test dataset to evaluate optimized model,"
            " outside Optuna Hyperparameter search."
            " Can be used if splitting strategy is set to NoSplitting."
            " Can be left empty if splitting strategy is set to split the data.",
        ),
    )
    save_intermediate_files: bool = field(
        default=False, metadata=schema(title="Save intermediate data files")
    )
    intermediate_training_dataset_file: Optional[str] = field(
        default=None,
        metadata=schema(title="Output file name for final training dataset file"),
    )
    intermediate_test_dataset_file: Optional[str] = field(
        default=None,
        metadata=schema(title="Output file name for final test dataset file"),
    )

    _train_smiles: List[str] = field(init=False, repr=False, metadata=skip)
    _train_y: np.ndarray = field(init=False, repr=False, metadata=skip)
    _test_smiles: List[str] = field(init=False, repr=False, metadata=skip)
    _test_y: np.ndarray = field(init=False, repr=False, metadata=skip)
    _sets_initialized: bool = field(default=False, init=False, metadata=skip)

    def _initialize_sets(self):
        train_smiles, train_y = read_data(
            self.training_dataset_file, self.input_column, self.response_column
        )
        test_smiles, test_y = read_data(
            self.test_dataset_file, self.input_column, self.response_column
        )

        train_smiles, train_y = deduplicate(
            train_smiles, train_y, self.deduplication_strategy
        )

        train_smiles, train_y, extra_test_smiles, extra_test_y = split(
            train_smiles, train_y, self.split_strategy
        )
        test_smiles, test_y = merge(
            extra_test_smiles, extra_test_y, test_smiles, test_y
        )

        self._train_smiles = train_smiles
        self._train_y = train_y
        self._test_smiles = test_smiles
        self._test_y = test_y

    def get_sets(self) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        """Returns training and test datasets."""

        if not self._sets_initialized:
            self._initialize_sets()
        return self._train_smiles, self._train_y, self._test_smiles, self._test_y

    def get_merged_sets(self) -> Tuple[List[str], np.ndarray]:
        """Returns merged training+test datasets."""

        train_smiles, train_y, test_smiles, test_y = self.get_sets()
        if test_smiles is not None and len(test_smiles) > 0:
            return merge(train_smiles, train_y, test_smiles, test_y)
        else:
            logger.warning(
                "Requested merged (training+test) dataset, "
                "but no test set specified, returning training dataset."
            )
            return train_smiles, train_y
