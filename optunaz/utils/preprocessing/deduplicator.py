import abc
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Literal

import pandas as pd
from apischema import deserializer, serializer, schema, identity
from apischema.conversions import Conversion
from apischema.metadata import none_as_undefined


class Deduplicator:
    """Base class for deduplicators.

    Each deduplicator should provide method `dedup`,
    which takes dataframe and name of SMILES column,
    and returns dataframe with duplicates removed.
    """

    # https://wyfo.github.io/apischema/0.17/examples/subclass_union/
    _union: Any = None

    # You can use __init_subclass__ to register new subclass automatically
    def __init_subclass__(cls, **kwargs):
        # Deserializers stack directly as a Union
        deserializer(Conversion(identity, source=cls, target=Deduplicator))
        # Only Base serializer must be registered (and updated for each subclass) as
        # a Union, and not be inherited
        Deduplicator._union = (
            cls if Deduplicator._union is None else Union[Deduplicator._union, cls]
        )
        serializer(
            Conversion(
                identity,
                source=Deduplicator,
                target=Deduplicator._union,
                inherited=False,
            )
        )

    @abc.abstractmethod
    def dedup(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        pass


@dataclass
class KeepFirst(Deduplicator):
    """Keep first."""

    name: Literal["KeepFirst"] = "KeepFirst"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        return df.drop_duplicates(subset=smiles_col, keep="first")


@dataclass
class KeepLast(Deduplicator):
    """Keep last."""

    name: Literal["KeepLast"] = "KeepLast"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        return df.drop_duplicates(subset=smiles_col, keep="last")


@dataclass
class KeepRandom(Deduplicator):
    """Keep random."""

    name: Literal["KeepRandom"] = "KeepRandom"
    seed: Optional[int] = field(
        default=None,
        metadata=schema(title="Seed for random number generator") | none_as_undefined,
    )

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        # https://stackoverflow.com/a/41650846
        return df.groupby(smiles_col, as_index=False).sample(
            n=1, random_state=self.seed
        )


@dataclass
class KeepMin(Deduplicator):
    """Keep min."""

    name: Literal["KeepMin"] = "KeepMin"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        # https://stackoverflow.com/a/41650846
        return df.groupby(smiles_col, as_index=False).min(numeric_only=True)


@dataclass
class KeepMax(Deduplicator):
    """Keep max."""

    name: Literal["KeepMax"] = "KeepMax"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        # https://stackoverflow.com/a/41650846
        return df.groupby(smiles_col, as_index=False).max(numeric_only=True)


@dataclass
class KeepAvg(Deduplicator):
    """Keep average. Classification will threshold at 0.5.

    This deduplicator converts input SMILES to canonical SMILES.
    """

    name: Literal["KeepAvg"] = "KeepAvg"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        """For regression, keep mean value."""

        # When de-duplicating by canonical SMILES,
        # original non-canonical SMILES is lost.
        # Both mean() and median() have this behavior.
        return df.groupby(smiles_col, as_index=False).mean(numeric_only=True)


@dataclass
class KeepMedian(Deduplicator):
    """Keep median. Classification will threshold at 0.5.

    This deduplicator converts input SMILES to canonical SMILES.
    """

    name: Literal["KeepMedian"] = "KeepMedian"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        """For regression, keep median value."""

        # When de-duplicating by canonical SMILES,
        # original non-canonical SMILES is lost.
        # Both mean() and median() have this behavior.
        return df.groupby(smiles_col, as_index=False).median(numeric_only=True)


@dataclass
class KeepAllNoDeduplication(Deduplicator):
    """Keep all.

    Do not perform any deduplication.
    """

    name: Literal["KeepAllNoDeduplication"] = "KeepAllNoDeduplication"

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        return df
