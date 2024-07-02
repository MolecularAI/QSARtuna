import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
from apischema import schema
from apischema.metadata import skip, none_as_undefined

from optunaz.utils import load_df_from_file
from optunaz.utils.preprocessing.deduplicator import (
    Deduplicator,
    KeepMedian,
)
from optunaz.utils.preprocessing.splitter import (
    NoSplitting,
    Splitter,
    GroupingSplitter,
    AnyInputDataSplitter,
)

from optunaz.descriptors import CanonicalSmiles, ValidDescriptor, descriptor_from_config

from optunaz.utils.preprocessing.transform import (
    LogBase,
    LogNegative,
    AnyAuxTransformer,
    DataTransform,
)

logger = logging.getLogger(__name__)


def isvalid(smiles: list[str]) -> np.ndarray:
    # returns 0/1 if input is None (for invalid MolFromSmiles molecules) or valid, respectively
    return np.array(
        descriptor_from_config(smiles, ValidDescriptor.new(), return_failed_idx=False),
        dtype=bool,
    )


def read_data(
    filename: Union[Path, str],
    smiles_col: str = "smiles",
    resp_col: str = None,
    response_type: str = None,
    aux_col: str = None,
    split_strategy: Splitter = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Reads data, drops NaNs and invalid SMILES.
    Supports SDF and CSV formats.
    In case of SDF - only response column has to be provided,
    since the smiles will be parsed from mol files inside.

    Returns a tuple of ( SMILES (X), responses (Y), groups (groups) ).
    """

    if filename is None:
        return [], np.empty((0,)), np.empty((0,)), np.empty((0,))

    df = load_df_from_file(filename, smiles_col)

    # Get labelled groups if the splitting strategy supports/requires this
    if hasattr(split_strategy, "groups"):
        groups = split_strategy.groups(df, smiles_col)

    # Remove rows, that hold "NaN"s in the respective columns of interest
    nrow_before = df.shape[0]
    if resp_col is not None:
        if aux_col is not None:
            df = df[[smiles_col, resp_col, aux_col]].dropna()
        else:
            df = df[[smiles_col, resp_col]].dropna()
    else:
        df = df[[smiles_col]].dropna()
    if df.shape[0] < nrow_before:
        logger.info(
            f"Removed {nrow_before - df.shape[0]} rows with NaNs during load and splitting steps."
        )
        if df.shape[0] == 0:
            raise ValueError("No compounds remain for modelling")

    dtype = np.float64  # default the dtype to numerical float for now
    # detect rows with to_numeric/to_boolean errors, report and fail the run
    if None not in (resp_col, response_type):
        row_issues = []
        if response_type == "classification":
            dtype = np.uint8
            df.loc[:, resp_col] = df[resp_col].map(
                {
                    True: 1,
                    False: 0,
                    "True": 1,
                    "False": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "1.0": 1,
                    "0.0": 0,
                }
            )
            row_issues = df[df[resp_col].isna().values]
        elif response_type == "regression":
            row_issues = df[
                df[[resp_col]].apply(pd.to_numeric, errors="coerce").isna().values
            ]
        if len(row_issues) > 0:
            row_issues = row_issues.reset_index().rename(
                {"index": "line_number"}, axis=1
            )
            row_issues.loc[:, "line_number"] = row_issues.loc[:, "line_number"] + 1
            raise ValueError(
                f"Non-numeric/boolean response in the following response column data:\n{row_issues}",
                row_issues,
            )

    # replace reaction ">>" here
    df[smiles_col] = df[smiles_col].str.replace(">>", ".")
    # Drop invalid. For 'index' trick, see https://stackoverflow.com/a/27360130.
    smiles = df[smiles_col]
    dropmask = ~isvalid(smiles)
    if dropmask.all():
        msg = f"No valid SMILES in user input column: {smiles_col}"
        logger.info(msg)
        raise ValueError(msg)
    if isinstance(split_strategy, GroupingSplitter):
        msg = (
            f"Split strategy {split_strategy} will drop "
            f"{df[smiles_col].map(groups).isna().sum()} NaN labelled rows"
        )
        logger.info(msg)
        dropmask = (dropmask | df[smiles_col].map(groups).isna()).values
        if dropmask.all():
            logger.info(f"No valid split labels for split strategy {split_strategy}")
            raise ValueError(msg)

    df.drop(df[dropmask].index, inplace=True)

    valid_smiles = df[smiles_col].to_list()
    valid_responses = df[resp_col].to_numpy(dtype=dtype) if resp_col else np.empty((0,))
    if aux_col:
        valid_auxs = df[aux_col] if aux_col else np.empty((0,))
    else:
        valid_auxs = None
    if isinstance(split_strategy, GroupingSplitter):
        valid_groups = df[smiles_col].map(groups).to_list()
    else:
        valid_groups = None

    return valid_smiles, valid_responses, valid_auxs, valid_groups


def deduplicate(
    smiles: List[str],
    y: np.ndarray,
    aux: np.ndarray,
    groups: np.ndarray,
    deduplication_strategy: Deduplicator,
    response_type: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Removes duplicates based on RDKit canonical SMILES representation.

    Returns a 2-tuple of original SMILES and deduplicated values.

    In case there is an ambiguity which SMILES to return,
    as is the case for deduplication by averaging,
    returns canonical SMILES instead.
    """

    smicol = "SMILES"
    cancol = "canonical_SMILES"
    respcol = "response"
    auxcol = "auxiliary"
    groupcol = "group"

    logger.info("Canonicalising SMILES")

    can_smiles = descriptor_from_config(
        smiles, CanonicalSmiles.new(), return_failed_idx=False
    )
    df = pd.DataFrame(
        {smicol: smiles, cancol: can_smiles, respcol: y, auxcol: aux, groupcol: groups}
    ).fillna(value=np.nan)
    failed_can = df.loc[df.canonical_SMILES.isna()]
    if failed_can.shape[0] > 0:
        df = df.dropna(subset=[cancol])
        logger.info(
            f"Deduplication canonicalization failed for {failed_can.shape[0]} molecules: {failed_can}"
        )

    df_group_map = df.set_index(cancol)[groupcol].to_dict()

    if df[auxcol].isnull().all():
        df_dedup = deduplication_strategy.dedup(df, cancol)
    else:
        # If auxiliary column is used then an ID from canonical SMILES and aux data tuple is created
        df["id"] = df[cancol].astype(str) + "|" + df[auxcol].astype(str)
        df_dedup = deduplication_strategy.dedup(df, "id")
        df_dedup.loc[:, cancol] = df_dedup.loc[:, "id"].str.split("|").str[0]
        df_dedup.loc[:, auxcol] = df_dedup.loc[:, "id"].str.split("|").str[1]
    if df_dedup.shape[0] < df.shape[0]:
        logger.info(f"Deduplication removed {df.shape[0] - df_dedup.shape[0]} rows")
    df = df_dedup.reset_index(drop=True)

    # Deduplication by averaging/median drops original SMILES column `smicol`.
    # In case `smicol` is missing, use `cancol`.
    dedup_smiles = df[smicol] if smicol in df.columns else df[cancol]

    dedup_y = df[respcol]
    dedup_aux = df[auxcol]

    # deduplication avg/median thresholding done here
    if response_type == "classification":
        logger.info(
            f"{dedup_y[(dedup_y>0)&(dedup_y<1)].shape[0]} comps will be thresholded \
             (majority vote) during deduplication, class counts: {dedup_y.value_counts()}"
        )
        dedup_y = dedup_y >= 0.5

    # Deduplication by averaging/median drops groups, which are added back here.
    dedup_groups = df[cancol].map(df_group_map)

    # If groups or aux are None, they are converted as expected by sklearn.
    if dedup_groups.isnull().all():
        dedup_groups = None
    if dedup_aux.isnull().all():
        dedup_aux = None

    return dedup_smiles, dedup_y, dedup_aux, dedup_groups


def split(
    X: List[str],
    y: np.ndarray,
    aux: np.ndarray,
    strategy: Splitter,
    groups: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    train, test = strategy.split(X, y, groups=groups)

    train_smiles = np.array(X)[train].tolist()  # List to numpy (for indexing) and back.
    if aux is not None:
        train_aux = np.array(aux)[train].tolist()
        extra_test_aux = np.array(aux)[test].tolist()
    else:
        train_aux = None
        extra_test_aux = None
    train_y = y[train]
    extra_test_smiles = np.array(X)[test].tolist()
    extra_test_y = y[test]

    return (
        train_smiles,
        train_y,
        train_aux,
        extra_test_smiles,
        extra_test_y,
        extra_test_aux,
    )


def merge(
    train_smiles: List[str],
    train_y: np.ndarray,
    train_aux: np.ndarray,
    test_smiles: List[str],
    test_y: np.ndarray,
    test_aux: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    merged_smiles = train_smiles + test_smiles
    merged_y = np.concatenate((train_y, test_y), axis=None)
    if isinstance(train_aux, np.ndarray) and len(train_aux.shape) == 2:
        merged_aux = np.concatenate((train_aux, test_aux), axis=0)
    else:
        merged_aux = np.concatenate((train_aux, test_aux), axis=None)
    try:
        if not any(merged_aux):
            merged_aux = None
    except ValueError:
        pass

    return merged_smiles, merged_y, merged_aux


def transform(
    smiles_: List[str],
    y_: np.ndarray,
    aux_: np.ndarray,
    transform: DataTransform,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    tansformed_y = transform.transform(y_)
    try:
        mask = np.isfinite(tansformed_y).values
    except AttributeError:
        mask = np.isfinite(tansformed_y)
    if len(mask) > 0 and (~mask).all():
        msg = (
            f"Transform settings incompatible with input data. Check your configuration"
        )
        logger.info(msg)
        raise ValueError(msg)
    if sum(~mask) > 0:
        logger.info(
            f"Transform {transform} will remove {sum(~mask)} incompatible data points."
        )
        smiles_ = np.array(smiles_)[mask].tolist()
        y_ = tansformed_y[mask]
        if aux_ is not None:
            aux_ = aux_[mask]
    else:
        y_ = tansformed_y
    return smiles_, y_, aux_


@dataclass
class Dataset:
    """Dataset.

    Holds training data, optional test data, and names of input and response columns.
    """

    training_dataset_file: str = field(
        metadata=schema(
            title="Main dataset file",
            description="This is training dataset if no splitting stratetgy is selected. If splitting strategy is "
            "selected, this dataset is split into training dataset and test dataset.",
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
    response_type: str = field(
        default=None,
        metadata=schema(
            title="Response type",
            description="Type of response column with the float/binary value that the model will try to predict.",
        )
        | none_as_undefined,
    )
    aux_column: Optional[str] = field(
        default=None,
        metadata=schema(
            title="Auxiliary descriptor column",
            description="Name of auxiliary column with additional values that the model use to predict.",
        )
        | none_as_undefined,
    )
    aux_transform: Optional[AnyAuxTransformer] = field(
        default=None,
        metadata=schema(
            title="Auxiliary data transform",
            description="Transform to apply to the descriptor aux column.",
        )
        | none_as_undefined,
    )
    deduplication_strategy: Deduplicator = field(
        default_factory=KeepMedian,
        metadata=schema(
            title="Deduplication strategy",
            description="What to do with duplicate annotations or replicates,"
            " for example if the same compounds have multiple measurements."
            " Stereoisomers also become duplicates,"
            " as algorithms we use do not support stereochemistry.",
        ),
    )
    split_strategy: AnyInputDataSplitter = field(
        default_factory=NoSplitting,
        metadata=schema(
            title="Evaluation split strategy",
            description="Split strategy for input data (n=1). Test set performance is calculated using a model "
            "trained with the train set and the optimal parameters identified for the train set. If No Splitting is "
            "selected, optimization is performed on a split applied to all input data and no test score "
            "is calculated.",
        ),
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
        metadata=schema(title="Output file name for final training dataset file")
        | none_as_undefined,
    )
    intermediate_test_dataset_file: Optional[str] = field(
        default=None,
        metadata=schema(title="Output file name for final test dataset file")
        | none_as_undefined,
    )
    log_transform: Optional[bool] = field(
        default=False,
        metadata=schema(
            title="Perform log transform on input data",
            description="Transforms user inputs to a log scale. Reverse transform performed at inference "
            "(i.e. outputs  inverse logged)",
        ),
    )
    log_transform_base: Optional[LogBase] = field(
        default=None,
        metadata=schema(
            title="Log transform base",
            description="Define the `log2`, `log10` or normal `log` for input data",
        ),
    )
    log_transform_negative: Optional[LogNegative] = field(
        default=None,
        metadata=schema(
            title="Log transform negation",
            description="Whether to use a negative logarithmic scale",
        ),
    )
    log_transform_unit_conversion: Optional[int] = field(
        default=None,
        metadata=schema(
            title="Log transform unit conversion",
            description="Whether to divide units by a power of 10^X, where X is the value selected here."
            "e.g. value=6 would be convert user input from micromolar to molar, for pXC50 calculations "
            "combined with -Log10 calculations. This is deactivated by leaving blank or `None`.",
        ),
    )
    probabilistic_threshold_representation: bool = field(
        default=False,
        metadata=schema(
            title="Perform probabilistic threshold transform",
            description="Probabilistic threshold representation (PTR) transforms regression user inputs, to account "
            "for the experimental uncertainty. Output is a regression scale. See the docs for details",
        ),
    )
    probabilistic_threshold_representation_threshold: Optional[float] = field(
        default=None,
        metadata=schema(
            title="Probabilistic representation decision threshold",
            description="User-defined threshold representing the decision boundary of relevance for molecule "
            "property. This becomes the 0.5 midpoint on the transformed scale.",
        ),
    )
    probabilistic_threshold_representation_std: Optional[float] = field(
        default=None,
        metadata=schema(
            title="Probabilistic representation standard deviation",
            description="The experimental reproducibility of input data. Larger values will increase the uncertainty "
            "near the representation threshold or decision boundary",
        ),
    )

    _train_smiles: List[str] = field(init=False, repr=False, metadata=skip)
    _train_y: np.ndarray = field(init=False, repr=False, metadata=skip)
    _train_aux: np.ndarray = field(init=False, repr=False, metadata=skip)
    _test_smiles: List[str] = field(init=False, repr=False, metadata=skip)
    _test_y: np.ndarray = field(init=False, repr=False, metadata=skip)
    _test_aux: np.ndarray = field(init=False, repr=False, metadata=skip)
    _sets_initialized: bool = field(default=False, init=False, metadata=skip)

    def _initialize_sets(self):
        train_smiles, train_y, train_auxs, train_groups = read_data(
            self.training_dataset_file,
            self.input_column,
            self.response_column,
            self.response_type,
            self.aux_column,
            self.split_strategy,
        )

        test_smiles, test_y, test_auxs, _ = read_data(
            self.test_dataset_file,
            self.input_column,
            self.response_column,
            self.response_type,
            self.aux_column,
        )

        train_smiles, train_y, train_auxs, train_groups = deduplicate(
            train_smiles,
            train_y,
            train_auxs,
            train_groups,
            self.deduplication_strategy,
            self.response_type,
        )

        (
            train_smiles,
            train_y,
            train_auxs,
            extra_test_smiles,
            extra_test_y,
            extra_test_aux,
        ) = split(train_smiles, train_y, train_auxs, self.split_strategy, train_groups)

        test_smiles, test_y, test_auxs = merge(
            extra_test_smiles,
            extra_test_y,
            extra_test_aux,
            test_smiles,
            test_y,
            test_auxs,
        )

        ptr_threshold = self.probabilistic_threshold_representation_threshold
        ptr_std = self.probabilistic_threshold_representation_std
        if self.log_transform:
            from optunaz.utils.preprocessing.transform import ModelDataTransform

            log_transform = ModelDataTransform.new(
                base=self.log_transform_base,
                negation=self.log_transform_negative,
                conversion=self.log_transform_unit_conversion,
            )
            train_smiles, train_y, train_auxs = transform(
                train_smiles, train_y, train_auxs, log_transform
            )
            test_smiles, test_y, test_auxs = transform(
                test_smiles, test_y, test_auxs, log_transform
            )
            if self.probabilistic_threshold_representation:
                ptr_threshold = log_transform.transform(ptr_threshold)
                ptr_std = log_transform.transform(ptr_std)
            logger.info("Log transform applied")

        if self.probabilistic_threshold_representation:
            from optunaz.utils.preprocessing.transform import PTRTransform

            ptr_transform = PTRTransform.new(threshold=ptr_threshold, std=ptr_std)
            train_smiles, train_y, train_auxs = transform(
                train_smiles, train_y, train_auxs, ptr_transform
            )
            test_smiles, test_y, test_auxs = transform(
                test_smiles, test_y, test_auxs, ptr_transform
            )
            logger.info("PTR transform applied")

        if self.aux_column and self.aux_transform is not None:
            if train_auxs is not None:
                train_auxs = self.aux_transform.transform(train_auxs)
                logger.info(
                    f"Transformed train auxiliary data with: {self.aux_transform.name}"
                )

            if test_auxs is not None:
                test_auxs = self.aux_transform.transform(test_auxs)
                logger.info(
                    f"Transformed test auxiliary data with: {self.aux_transform.name}"
                )

        if self.save_intermediate_files:
            logger.info("Saving intermediate files")
            pd.DataFrame(
                data={
                    "train_smiles": train_smiles,
                    "train_y": train_y,
                    "train_aux": train_auxs,
                }
            ).to_csv(self.intermediate_training_dataset_file, index=False)
            pd.DataFrame(
                data={
                    "test_smiles": test_smiles,
                    "test_y": test_y,
                    "test_aux": test_auxs,
                }
            ).to_csv(self.intermediate_test_dataset_file, index=False)

        self._train_smiles = train_smiles
        self._train_y = train_y
        self._train_aux = train_auxs
        self._test_smiles = test_smiles
        self._test_y = test_y
        self._test_aux = test_auxs
        self._sets_initialized = True

        logger.info(f"Initialized sets")
        logger.info(f"len(_train_smiles):{len(train_smiles)}")
        logger.info(f"len(_train_y):{len(train_y)}")
        try:
            logger.info(f"len(_train_auxs):{len(train_auxs)}")
        except TypeError:
            logger.info(f"len(_train_auxs): 0")
        logger.info(f"len(_test_smiles):{len(test_smiles)}")
        logger.info(f"len(_test_y):{len(test_y)}")
        try:
            logger.info(f"len(_test_auxs):{len(test_auxs)}")
        except TypeError:
            logger.info(f"len(_test_auxs): 0")

    def get_sets(
        self,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
        """Returns training and test datasets."""

        if not self._sets_initialized:
            self._initialize_sets()
        self.check_sets()
        return (
            self._train_smiles,
            self._train_y,
            self._train_aux,
            self._test_smiles,
            self._test_y,
            self._test_aux,
        )

    def get_merged_sets(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Returns merged training+test datasets."""

        (
            train_smiles,
            train_y,
            train_aux,
            test_smiles,
            test_y,
            test_aux,
        ) = self.get_sets()
        if test_smiles is not None and len(test_smiles) > 0:
            return merge(
                train_smiles, train_y, train_aux, test_smiles, test_y, test_aux
            )
        else:
            logger.warning(
                "Requested merged (training+test) dataset, "
                "but no test set specified, returning training dataset."
            )
            return train_smiles, train_y, train_aux

    def check_sets(self):
        """Check sets are valid"""

        assert (
            len(self._train_smiles) > 0
        ), f"Insufficient train SMILES ({len(self._train_smiles)}), check data and/or splitting strategy"
        if self.response_type == "classification":
            assert (
                len(set(self._train_y)) == 2
            ), f"Train is not binary classification ({len(set(self._train_y))} distinct values), check data and/or splitting strategy"
        elif self.response_type == "regression":
            assert (
                len(set(self._train_y)) >= 5
            ), f"Train values do not appear valid for regression ({len(set(self._train_y))} response values)"
        if self._test_smiles is not None and len(self._test_smiles) > 0:
            if self.response_type == "classification":
                assert (
                    len(set(self._test_y)) == 2
                ), f"Test is not binary classification ({len(set(self._test_y))} distinct values), check data and/or splitting strategy"
            elif self.response_type == "regression":
                assert (
                    len(set(self._test_y)) >= 5
                ), f"Train values do not appear valid for regression ({len(set(self._test_y))} response values)"
