from dataclasses import dataclass
from typing import Optional, Union, Any

import pandas as pd
from apischema import deserializer, serializer
from apischema.conversions import Conversion, identity
from typing_extensions import Literal  # Python 3.7


class Deduplicator:
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


@dataclass
class KeepFirst(Deduplicator):
    name: Literal["KeepFirst"] = "KeepFirst"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        return df.drop_duplicates(subset=smiles_col, keep="first")


@dataclass
class KeepLast(Deduplicator):
    name: Literal["KeepLast"] = "KeepLast"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        return df.drop_duplicates(subset=smiles_col, keep="last")


@dataclass
class KeepRandom(Deduplicator):
    name: Literal["KeepRandom"] = "KeepRandom"
    seed: Optional[int] = None

    def dedup(self, df: pd.DataFrame, smiles_col: str):
        # Shuffle, then keep first.

        # Shuffle: https://stackoverflow.com/a/34879805
        shuffled = df.sample(frac=1, random_state=self.seed)

        return KeepFirst.dedup(shuffled, smiles_col)


@dataclass
class KeepMin(Deduplicator):
    name: Literal["KeepMin"] = "KeepMin"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        # https://stackoverflow.com/a/41650846
        return df.groupby(smiles_col, as_index=False).min()


@dataclass
class KeepMax(Deduplicator):
    name: Literal["KeepMax"] = "KeepMax"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        # https://stackoverflow.com/a/41650846
        return df.groupby(smiles_col, as_index=False).max()


@dataclass
class KeepAvg(Deduplicator):
    name: Literal["KeepAvg"] = "KeepAvg"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        """For regression, keep mean value."""
        return df.groupby(smiles_col, as_index=False).mean()


@dataclass
class KeepAllNoDeduplication(Deduplicator):
    name: Literal["KeepAllNoDeduplication"] = "KeepAllNoDeduplication"

    @staticmethod
    def dedup(df: pd.DataFrame, smiles_col: str):
        return df
