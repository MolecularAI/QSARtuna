import abc
import warnings
from dataclasses import dataclass, field
from typing import Union, Any

import numpy as np
import pandas as pd
from apischema import deserializer, serializer
from apischema import schema
from apischema.conversions import Conversion, identity
from apischema.metadata import skip
from apischema.skip import Skip
from sklearn.model_selection import train_test_split
from typing_extensions import Literal, Annotated  # Python 3.7


def split_randomly(df: pd.DataFrame, fraction=0.2, seed=42):
    """Returns two randomly split datasets, the first with "1-fraction" and the second with
    "fraction" observations."""
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(
            "Parameter fraction must be a number between 0 and 1 (exclusively)."
        )

    # Note: new version of sklearn.model_selection.train_test_split is able to return pandas data frames
    training_set, test_set = train_test_split(
        df, shuffle=True, test_size=fraction, random_state=seed
    )
    return training_set, test_set


def split_temporal(df: pd.DataFrame, time_column: str, threshold: float):
    """Function that will return two temporally split datasets, the the first (up and including the threshold) is
    the training, the remaining observations constitute the test sets, respectively."""

    # set the type of the column to be numeric for ordering
    df[time_column] = df[time_column].astype("float")

    # split the sets
    training_set = df[df[time_column] <= threshold]
    test_set = df[df[time_column] > threshold]

    # do warn, if any of the sets are empty
    if len(training_set) == 0 or len(test_set) == 0:
        warnings.warn("One or both sets are empty.")

    return training_set, test_set


def split_stratified(
    df: pd.DataFrame, respCol: str, fraction=0.2, bins="fd", seed=None
):
    """Function that will return two stratified split datasets, i.e. the input distribution will be binned by the
    specification used and for each bin a fraction is sampled randomly. This ensures that both sets resemble
    a similar distribution."""

    # TODO: check that for classification np.histogram() is similar to StratifiedKfold.

    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(
            "Parameter fraction must be a number between 0 and 1 (exclusively)."
        )

    # bin the values and return two sets
    samples_per_bin, bins = np.histogram(df[respCol], bins=bins)

    # set the bins to be just a little larger at the ends to include every observation
    bins[0] = np.nextafter(bins[0], -np.inf)
    bins[-1] = np.nextafter(bins[-1], np.inf)

    # introduce safe-guard for cases, where too few observations were present in a bin
    bins = np.delete(bins, np.flatnonzero(samples_per_bin < 10))

    # calculate the indices (bin-IDs) for the response values
    bin_idxs = np.digitize(x=df[respCol], bins=bins)

    # split and return the sets
    train, test = train_test_split(
        df, stratify=bin_idxs, shuffle=True, test_size=fraction, random_state=seed
    )
    return train, test


class Splitter:
    _union: Annotated[Any, Skip] = None

    # You can use __init_subclass__ to register new subclass automatically
    def __init_subclass__(cls, **kwargs):
        # Deserializers stack directly as a Union
        deserializer(Conversion(identity, source=cls, target=Splitter))
        # Only Base serializer must be registered (and updated for each subclass) as
        # a Union, and not be inherited
        Splitter._union = (
            cls if Splitter._union is None else Union[Splitter._union, cls]
        )
        serializer(
            Conversion(
                identity, source=Splitter, target=Splitter._union, inherited=False
            )
        )

    @abc.abstractmethod
    def split(self, df: pd.DataFrame):
        pass


@dataclass
class Random(Splitter):
    """Random splitting strategy."""

    name: Literal["Random"] = "Random"
    fraction: float = field(
        default=0.2, metadata=schema(title="Fraction of samples to use for test set")
    )
    seed: int = field(
        default=1,
        metadata=schema(
            title="Random seed", description="Random seed, for repeatable splits."
        ),
    )

    def split(self, df: pd.DataFrame):
        return split_randomly(df, self.fraction, self.seed)


@dataclass
class Temporal(Splitter):
    """Temporal split. Assumes that data is sorted."""

    name: Literal["Temporal"] = "Temporal"
    fraction: float = field(
        default=0.2, metadata=schema(title="Fraction of samples to use for test set")
    )

    def split(self, df: pd.DataFrame):
        train_size = int(len(df) * (1.0 - self.fraction))
        train = df[0:train_size]
        test = df[train_size:]
        return train, test


@dataclass
class Stratified(Splitter):
    name: Literal["Stratified"] = "Stratified"
    fraction: float = field(
        default=0.2, metadata=schema(title="Fraction of samples to use for test set")
    )
    bins: str = "fd"
    seed: int = field(
        default=1,
        metadata=schema(
            title="Random seed", description="Random seed, for repeatable splits."
        ),
    )
    respcol: str = field(default="response", metadata=skip)

    def split(self, df: pd.DataFrame):
        return split_stratified(
            df, respCol=self.respcol, fraction=self.fraction, bins=self.bins, seed=self.seed
        )


@dataclass
class NoSplitting(Splitter):
    name: Literal["NoSplitting"] = "NoSplitting"

    def split(self, df: pd.DataFrame):
        empty_test = df[0:0]
        return df, empty_test
