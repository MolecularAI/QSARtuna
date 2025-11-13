import abc
from dataclasses import dataclass, field
from typing import Dict, Iterator, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from apischema import schema
from apischema.metadata import none_as_undefined
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    BaseCrossValidator,
    PredefinedSplit,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from skmultilearn.model_selection import IterativeStratification


class SklearnSplitter(abc.ABC):
    """Interface definition for scikit-learn cross-validation splitter.

    Scikit-learn does not define a class that describes the splitter interface.
    Instead, scikit-learn describes in text that splitter should have two methods:
    'get_n_splits' and 'split'.

    This class describes this splitter interface as an abstract Python class,
    for convenience and better type checking.
    """

    @abc.abstractmethod
    def get_n_splits(self, X, y, groups) -> int:
        pass

    @abc.abstractmethod
    def split(self, X, y, groups) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        pass


class Splitter:
    """Splitter for input data.

    This is the base class for classes that split input data into train and test.

    See also CvSplitter for making multiple cross-validation splits.

    Splitter and CvSplitter are used to define valid input choices
    for splitting data into train-test sets,
    and for splitting train data into cross-validation splits
    in scikit-learn cross_validate function.
    These two sets of options might be different
    (although underlying implementations might be merged).
    """

    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """Splits input and returns indices for train and test sets.

        Returns two numpy arrays:
        one with indices of train set,
        and one with indices of test set.

        Note that scikit-learn splitters return an Iterator
        that yields (train, test) tuples for multiple splits,
        here we return only one split.
        """
        # Default impl:
        cv = self.get_sklearn_splitter(n_splits=1)
        iterator = cv.split(X, y, groups)
        first_split = next(iterator)
        return first_split

    @abc.abstractmethod
    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        pass


@dataclass
class Random(Splitter):
    """Random split."""

    name: Literal["Random"] = "Random"
    fraction: float = field(
        default=0.2,
        metadata=schema(
            title="Fraction samples",
            description="Fraction of samples to use for test set.",
            min=0.0,
        )
        | none_as_undefined,
    )
    seed: Optional[int] = field(
        default=1,
        metadata=schema(
            title="Seed for random number generator",
            description="Seed for random number generator, for repeatable splits.",
        )
        | none_as_undefined,
    )
    leave_out: Optional[float] = field(
        default=0.0,
        metadata=schema(
            title="Leave out fraction",
            description="Fraction of samples that will not be used in train or test set, to reduce compute time.",
            min=0.0,
        )
        | none_as_undefined,
    )

    def get_sklearn_splitter(self, n_splits: int) -> ShuffleSplit:
        if self.leave_out is not None:
            train_size = (1 - self.fraction) - self.leave_out
            assert (
                train_size > 0.0
            ), f"not possible to leave out {self.leave_out}, since no train remains given {self.fraction} test fraction"
        else:
            train_size = None
        return ShuffleSplit(
            n_splits=n_splits,
            test_size=self.fraction,
            train_size=train_size,
            random_state=self.seed,
        )


@dataclass
class Temporal(Splitter):
    """Temporal split.

    When used for performance evaluation Temporal assumes that the data is sorted,
    with the oldest entries in the beginning of the file and the newest entries added at the end.

    When used for optimization cross validation, Temporal splitting uses a TimeSeriesSplit from Sklearn.
    This ensures successive hyperparameter sets are supersets of those that come before them.
    """

    name: Literal["Temporal"] = "Temporal"
    fraction: float = field(
        default=0.2, metadata=schema(title="Fraction of samples to use for test set")
    )

    def split(self, X, y=None, groups=None):
        train_size = int(len(X) * (1.0 - self.fraction))
        train = np.arange(0, train_size)
        test = np.arange(train_size, len(X))
        return train, test

    def get_sklearn_splitter(self, n_splits: int) -> TimeSeriesSplit:
        return TimeSeriesSplit(n_splits=n_splits)


@dataclass
class Stratified(Splitter):
    """Real-valued Stratified Shuffle Split.

    This is similar to scikit-learn StratifiedShuffleSplit,
    but uses histogram binning for real-valued inputs.

    If inputs are integers (or strings),
    this splitter reverts to StratifiedShuffleSplit.
    """

    name: Literal["Stratified"] = "Stratified"
    fraction: float = field(
        default=0.2,
        metadata=schema(
            title="Fraction samples",
            description="Fraction of samples to use for test set.",
            min=0.0,
        )
        | none_as_undefined,
    )
    seed: Optional[int] = field(
        default=1,
        metadata=schema(
            title="Seed for random number generator",
            description="Seed for random number generator, for repeatable splits.",
        )
        | none_as_undefined,
    )
    leave_out: Optional[float] = field(
        default=0.0,
        metadata=schema(
            title="Leave out fraction",
            description="Fraction of samples that will not be used in train or test set, to reduce compute time.",
            min=0.0,
        )
        | none_as_undefined,
    )
    bins: str = field(
        default="fd_merge",
        metadata=schema(
            title="Binning algorithm",
            description="Algorithm to use for determining histogram bin edges,"
            " see numpy.histogram for possible options, or use default 'fd_merge'",
        ),
    )

    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        if self.leave_out is not None:
            train_size = (1 - self.fraction) - self.leave_out
            assert (
                train_size > 0.0
            ), f"not possible to leave out {self.leave_out}, since no train remains given {self.fraction} test fraction"
        else:
            train_size = None
        return HistogramStratifiedShuffleSplit(
            n_splits=n_splits,
            test_fraction=self.fraction,
            bins=self.bins,
            random_state=self.seed,
            train_size=train_size,
        )


@dataclass
class NoSplitting(Splitter):
    """No splitting.

    Do not perform any splitting.
    Returns all input data as training set,
    and returns an empty test set.
    """

    name: Literal["NoSplitting"] = "NoSplitting"

    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        train = np.arange(0, len(X))
        test = np.array([], dtype=int)  # Empty.
        return train, test

    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        raise NotImplementedError()


@dataclass
class KFold(Splitter):
    """KFold.

    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation,
    while the k - 1 remaining folds form the training set.
    """

    name: Literal["KFold"] = "KFold"
    shuffle: bool = field(
        default=True,
        metadata=schema(
            title="Shuffle",
            description="Whether to shuffle the data before splitting into batches."
            " Note that the samples within each split will not be shuffled.",
        ),
    )
    random_state: Optional[int] = field(
        default=1,
        metadata=schema(
            title="Random state",
            description="When shuffle is True,"
            " random_state affects the ordering of the indices,"
            " which controls the randomness of each fold."
            " Otherwise, this parameter has no effect."
            " Pass an int for reproducible output across multiple function calls.",
        ),
    )

    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_sklearn_splitter(self, n_splits: int) -> sklearn.model_selection.KFold:
        return sklearn.model_selection.KFold(
            n_splits=n_splits, shuffle=self.shuffle, random_state=self.random_state
        )


# rectify error with numpy trying to allocate too large linspace
def fd_bin(y: np.ndarray) -> np.ndarray:
    """Empty bin merging histogram based on:
    https://github.com/numpy/numpy/issues/11879 and
    https://github.com/numpy/numpy/issues/10297

    The modification avoids this via merging adjacent empty bins"""
    a_unsorted = np.array(y)
    left_cap, right_cap = a_unsorted.min(), a_unsorted.max()
    a = np.sort(a_unsorted) - left_cap
    fd = np.lib.histograms._hist_bin_fd(a, range)
    left_edges = a // fd * fd
    right_edges = left_edges + fd
    new_bins = np.unique(np.concatenate((left_edges, right_edges))) + left_cap
    return np.append(new_bins, right_cap + fd)


def stratify(y: np.ndarray, bins: str = "fd") -> np.ndarray:
    """Stratifies (splits into groups) the values in 'y'.

    If input 'y' is real-valued (numpy.dtype.kind == 'f'),
    this function bins the values based on computed histogram edges.

    For all other types of inputs,
    this function returns the original array,
    since downstream algorithms can natively deal with integer and categorical data.
    """

    # Bin the values
    if bins == "fd_merge":
        # implement fd avoiding this issue: https://github.com/numpy/numpy/issues/11879
        samples_per_bin, bins = np.histogram(y, bins=fd_bin(y))
    else:
        samples_per_bin, bins = np.histogram(y, bins=bins)

    # Extend the first and the last bin by a tiny amount, to include every observation.
    bins[0] = np.nextafter(bins[0], -np.inf)
    bins[-1] = np.nextafter(bins[-1], np.inf)

    # Drop the bins with too few observations.
    bins = np.delete(bins, np.flatnonzero(samples_per_bin < 10))
    if samples_per_bin[0] <= 10:
        bins = np.delete(bins, 0)

    # Get the bin indices (bin-IDs) for each value.
    bin_idxs = np.digitize(x=y, bins=bins)

    return bin_idxs


@dataclass
class HistogramStratifiedShuffleSplit(SklearnSplitter):
    """HistogramStratifiedShuffleSplit

    StratifiedShuffleSplit for real-valued inputs.
    """

    # Backend/sklearn part.

    test_fraction: float = 0.1
    n_splits: int = 10
    bins: str = "fd_merge"
    random_state: Optional[int] = 42
    train_size: float = 0.0

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        # Here we stratify 'y' ourselves when it is floating-point ("np.inexact").
        # Then we delegate the actual splitting to StratifiedShuffleSplit (SSS).
        # If elements in y are integer or string, SSS handles them natively.
        if issubclass(y.dtype.type, np.inexact):
            y_sss = stratify(y, self.bins)
        else:
            y_sss = y

        sss = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_fraction,
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return sss.split(X, y_sss, groups)


class GroupingSplitter(Splitter, abc.ABC):
    """Splitter for methods using the group method

    This is the base class for the Predefined and ScaffoldSplit classes.
    """

    @abc.abstractmethod
    def groups(self, df, smiles_col) -> Dict:
        ...


@dataclass
class Predefined(GroupingSplitter):
    """Predefined split.

    Splits data based predefined labels in a column. Integers can be used, and `-1` flags datapoints for use only in
    the training set. Data points with missing (NaN) values will be removed from train or test
    """

    column_name: Optional[str] = field(
        default=None,
        metadata=schema(
            title="Column Name",
            description="Name of the column with labels for splits. Use `-1` to denote datapoints for the train set",
        )
        | none_as_undefined,
    )
    name: Literal["Predefined"] = "Predefined"

    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        raise NotImplementedError()

    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        assert groups is not None, "`groups` should be supplied for Predefined splitter"
        ps = PredefinedSplit(groups)
        try:
            return next(ps.split(X))
        except StopIteration:
            raise StopIteration(
                "Predefined split not valid, check configuration and data"
            )

    def groups(self, df, smiles_col) -> Dict:
        assert (
            self.column_name is not None
        ), "Predefined split should be supplied with a `column_name` with labels"
        groups = df.set_index(smiles_col)[self.column_name].dropna()
        # maintain the `-1` manually defined training set, if it is present
        if -1 in groups.unique():
            return groups.to_dict()
        # otherwise convert the users' column to a category code to ensure compatibility
        else:
            return groups.astype("category").cat.codes.to_dict()


def butina_cluster(groups, cutoff=0.4):
    """
    Clusters the scaffolds based on Butina and returns the scaffold grouping labels
    """
    from joblib import Parallel, delayed, effective_n_jobs
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    from optunaz.descriptors import ECFP, descriptor_from_config

    # deduplicate the scaffolds and generate fingerprints
    n_cores = effective_n_jobs(-1)
    distinct_smiles = groups.unique().tolist()
    fps = descriptor_from_config(
        distinct_smiles,
        ECFP.new(nBits=1024, radius=2, returnRdkit=True),
        return_failed_idx=False,
    )

    # butina cluster the fingerprints.
    # See https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html for details
    dists = Parallel(n_jobs=n_cores, prefer="threads")(
        delayed(DataStructs.BulkTanimotoSimilarity)(fps[i], fps[:i], returnDistance=1)
        for i in range(1, len(fps))
    )
    dists = np.concatenate(dists, axis=None)

    cs = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    distinct_groups = [0] * len(fps)
    for idx, cluster in enumerate(cs, 1):
        for member in cluster:
            distinct_groups[member] = idx

    # return unique list of the scaffold grouping as a dict
    group_dict = dict(zip(distinct_smiles, distinct_groups))
    return groups.map(group_dict)


@dataclass
class ScaffoldSplit(GroupingSplitter):
    """Stratified Group K Fold based on chemical scaffold.

    Splits data based chemical (Murcko) scaffolds for the compounds in the user input data.
    This emulates the real-world scenario when models are applied to novel chemical space
    """

    bins: str = field(
        default="fd_merge",
        metadata=schema(
            title="Binning algorithm",
            description="Algorithm to use for determining histogram bin edges,"
            " see numpy.histogram for possible options, or use default 'fd_merge'",
        ),
    )
    random_state: int = field(
        default=42,
        metadata=schema(
            title="Random State",
            description="Seed used to define which scaffold splits are used for training and testing",
        ),
    )
    make_scaffold_generic: bool = field(
        default=True,
        metadata=schema(
            title="Make scaffold generic",
            description="Makes Murcko scaffolds generic by removing hetero-atoms",
        ),
    )
    butina_cluster: float = field(
        default=0.4,
        metadata=schema(
            min=0.0,
            max=1.0,
            title="Cluster threshold",
            description="Butina clustering to aggregate scaffolds into shared folds. Elements within this "
            "cluster range are considered neighbors, increasing test difficulty. `0.0` turns Butina "
            "clustering off",
        ),
    )
    name: Literal["ScaffoldSplit"] = "ScaffoldSplit"

    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        """Returns an iterable that yields all splits from StratifiedGroupKFold."""

        class ScaffoldSplitSklearnWrapper(SklearnSplitter):
            def __init__(self, scaffold_split, n_splits):
                self.scaffold_split = scaffold_split
                self.n_splits = n_splits

            def get_n_splits(self, X=None, y=None, groups=None):
                return self.scaffold_split.get_n_splits(X, y, groups)

            def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
                assert groups is not None, (
                    "ScaffoldSplit expects scaffold groups supplied with the `split` function. This "
                    "can be assisted with the `groups` function of ScaffoldSplit. "
                )
                # Similar to Histogram split, deal with continuous or binary y
                y_sss = (
                    stratify(y, self.scaffold_split.bins)
                    if issubclass(y.dtype.type, np.inexact)
                    else y
                )
                # Butina cluster if distance is greater than 0
                if self.scaffold_split.butina_cluster > 0.0:
                    groups = butina_cluster(
                        groups, cutoff=self.scaffold_split.butina_cluster
                    )
                sgkf = StratifiedGroupKFold(
                    random_state=self.random_state,
                    shuffle=True,
                )
                return sgkf.split(X, y_sss, groups)

            def groups(self, smiles) -> Dict:
                df = pd.DataFrame(smiles, columns=["smiles"])
                scaffold_dict = self.scaffold_split.groups(df, "smiles")
                return df["smiles"].map(scaffold_dict)

        return ScaffoldSplitSklearnWrapper(self, n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        raise NotImplementedError()

    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        assert groups is not None, (
            "ScaffoldSplit expects scaffold groups supplied with the `split` function. This "
            "can be assisted with the `groups` function of ScaffoldSplit. "
        )
        # Similar to Histogram split, deal with continuous or binary y
        y_sss = stratify(y, self.bins) if issubclass(y.dtype.type, np.inexact) else y
        # Butina cluster if distance is greater than 0
        if self.butina_cluster > 0.0:
            groups = butina_cluster(groups, cutoff=self.butina_cluster)
        sgkf = StratifiedGroupKFold(
            random_state=self.random_state,
            shuffle=True,
        )
        return next(sgkf.split(X, y_sss, groups))

    def groups(self, df, smiles_col) -> Dict:
        """Calculate scaffold smiles from a smiles column"""
        from optunaz.descriptors import (
            GenericScaffold,
            Scaffold,
            descriptor_from_config,
        )

        if self.make_scaffold_generic:
            df["scaffold"] = descriptor_from_config(
                df[smiles_col], GenericScaffold.new(), return_failed_idx=False
            )
        else:
            df["scaffold"] = descriptor_from_config(
                df[smiles_col], Scaffold.new(), return_failed_idx=False
            )

        return df.set_index(smiles_col)["scaffold"].to_dict()


@dataclass
class StratifiedOverlappingGroupKFold(SklearnSplitter):
    """StratifiedOverlappingGroupKFold

    This splitter performs class-wise stratified K-Fold splitting while ensuring stratified _overlapping_ groups.
    It uses KMeans clustering on X followed by stratification to ensure splits balance classes and groups.

    The intention is to yield splits representative of both X and Y data for tasks like explainability
    where inference is costly. By using multi-label IterativeStratification function the folds preserve the
    percentage of samples for (real-value) stratified class and clustered groups based on X.
    """

    bins: str = field(
        default="fd_merge",
        metadata=schema(
            title="Binning algorithm",
            description="Algorithm to use for determining histogram bin edges for y values,"
            " see numpy.histogram for possible options, or use default 'fd_merge'",
        ),
    )
    random_state: int = field(
        default=42,
        metadata=schema(
            title="Random State",
            description="Seed used to define which scaffold splits are used for training and testing",
        ),
    )
    n_clusters: int = field(
        default=8,
        metadata=schema(
            min=1,
            title="Number of KMeans clusters",
            description="Specifies the number of clusters in the X space",
        ),
    )
    n_replicates: int = field(
        default=5,
        metadata=schema(
            min=1,
            title="Number of replicates",
            description="Specifies the number of replicates to perform",
        ),
    )
    fraction: float = field(
        default=0.2,
        metadata=schema(
            title="Fraction samples",
            description="Fraction of samples to use for test set",
            min=0.0,
        )
        | none_as_undefined,
    )

    name: Literal["StratifiedOverlappingGroupKFold"] = "StratifiedOverlappingGroupKFold"

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y = stratify(y, self.bins) if issubclass(y.dtype.type, np.inexact) else y

        """
        Yields the first split of the underlying cross-validator for each replicate,
        with a reproducible random seed for each replicate.
        """
        for replicate in range(self.n_replicates):
            # Generate a reproducible seed for each replicate
            random_state = self.random_state + replicate
            clusters = KMeans(
                n_clusters=self.n_clusters, random_state=random_state
            ).fit_predict(X)
            groups = np.column_stack((y, clusters))

            stratifier = IterativeStratification(
                n_splits=2,
                order=2,
                sample_distribution_per_fold=[self.fraction, 1.0 - self.fraction],
            )
            yield from stratifier.split(X, groups)


AnyCvSplitter = Union[
    Stratified, Random, ScaffoldSplit, Temporal, StratifiedOverlappingGroupKFold
]

AnyInputDataSplitter = Union[
    Random,
    Temporal,
    Stratified,
    NoSplitting,
    Predefined,
    ScaffoldSplit,
    StratifiedOverlappingGroupKFold,
]


@dataclass
class ReplicatedCrossValidator(BaseCrossValidator, SklearnSplitter):
    """ReplicatedCrossValidator

    Custom cross-validator that wraps an existing cross-validator and repeats
    the first split for a specified number of replicates, using reproducible seeds.
    """

    cv: AnyCvSplitter | SklearnSplitter = field(
        default=Random,
        metadata=schema(
            title="Cross-validation strategy",
            description="Underlying cross-validation strategy to use for the first split.",
        ),
    )
    random_state: int = field(
        default=42,
        metadata=schema(
            title="Random state",
            description="Ensures ReplicatedCrossValidator is reproducible across runs.",
        ),
    )
    n_replicates: int = field(
        default=5,
        metadata=schema(
            min=1,
            title="Number of replicates",
            description="Number of times to repeat the first split.",
        ),
    )
    n_splits: int = field(
        default=5,
        metadata=schema(
            min=1,
            title="Number of splits",
            description="Number of slits applied to the cv.",
        ),
    )
    name: Literal["ReplicatedCrossValidator"] = "ReplicatedCrossValidator"

    def get_sklearn_splitter(self, n_splits: int) -> SklearnSplitter:
        return self.cv.get_sklearn_splitter(n_splits)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Returns the total number of splits (equal to num_replicates).
        """
        return self.n_replicates

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields the first split of the underlying cross-validator for each replicate,
        with a reproducible random seed for each replicate.
        """
        for replicate in range(self.n_replicates):
            # Generate a reproducible seed for each replicate
            random_state = self.random_state + replicate
            if hasattr(self.cv, "get_sklearn_splitter"):
                cv = self.cv.get_sklearn_splitter(n_splits=self.n_splits)
            else:
                cv = self.cv
            cv.random_state = random_state

            # Get the first split from the underlying cross-validator
            yield next(cv.split(X, y, groups))
