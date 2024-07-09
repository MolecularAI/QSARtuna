import abc
import inspect
from dataclasses import field, dataclass
import apischema
import scipy
from scipy.stats import norm
from apischema import schema, deserializer, serializer, identity
from apischema.conversions import Conversion
from typing import Union, Any, Literal, Annotated, Optional
from enum import Enum
import pandas as pd
import numpy as np
from optunaz.config import NameParameterDataclass


class DataTransformError(Exception):
    """Raised when insufficient molecules for UnfittedSklearnSclaer to fit"""

    pass


class DataTransform(NameParameterDataclass, abc.ABC):
    """Base class for auxiliary transformers.

    Each data transformer should provide method `transform`,
    which takes raw input data, and returns numpy arrays with
    transformed output data.
    """

    _union: Any = None

    # You can use __init_subclass__ to register new subclass automatically
    def __init_subclass__(cls, **kwargs):
        if inspect.isabstract(cls):
            return  # Do not register abstract classes, like RDKitDescriptor.
        # Deserializers stack directly as a Union
        deserializer(Conversion(identity, source=cls, target=DataTransform))
        # Only Base serializer must be registered (and updated for each subclass) as
        # a Union, and not be inherited
        DataTransform._union = (
            cls if DataTransform._union is None else Union[DataTransform._union, cls]
        )
        serializer(
            Conversion(
                identity,
                source=DataTransform,
                target=DataTransform._union,
                inherited=False,
            )
        )

    @abc.abstractmethod
    def transform(self, y_: np.array) -> np.array:
        pass


@dataclass
class PTRTransform(DataTransform):
    """Transform model input/output with PTR"""

    @apischema.type_name("PTRTransformParams")
    @dataclass
    class Parameters:
        threshold: Annotated[
            float,
            schema(
                title="PTR Threshold",
                description="The decision boundary for discretising active or inactive classes used by PTR.",
            ),
        ] = field(
            default=None,
        )
        std: Annotated[
            float,
            schema(
                title="PTR standard deviation",
                description="The standard deviation used by PTR, e.g. experimental reproducibility/uncertainty",
            ),
        ] = field(
            default=None,
        )

    name: Literal["PTRTransform"] = "PTRTransform"
    parameters: Parameters = Parameters()

    def transform(self, y_) -> np.ndarray:
        assert self.parameters.threshold is not None, "Must define a PTR threshold"
        assert self.parameters.std is not None, "Must define a PTR Std. Dev."
        y_ = np.array(y_)
        mask = np.isfinite(y_)
        ret = np.zeros(y_.shape)
        ret[mask] = norm.cdf(
            np.array(y_[mask]).clip(
                0.0000000000000001,
            ),
            self.parameters.threshold,
            self.parameters.std,
        )
        ret[~mask] = np.nan
        return ret

    def reverse_transform(self, y_) -> np.ndarray:
        assert self.parameters.threshold is not None, "Must define a PTR threshold"
        assert float(self.parameters.threshold) != float(
            "nan"
        ), "PTR threshold must not be nan"
        assert self.parameters.std is not None, "Must define a PTR Std. Dev."
        y_ = np.array(y_)
        mask = np.isfinite(y_)
        ret = np.zeros(y_.shape)
        ret[mask] = norm.ppf(
            y_[mask].clip(0.0000000000000001, 0.9999999999999999),
            self.parameters.threshold,
            self.parameters.std,
        )
        ret[~mask] = np.nan
        return ret


class LogBase(str, Enum):
    """Base for Numpy transform in ModelDataTransform"""

    LOG2 = "log2"
    LOG10 = "log10"
    LOG = "log"


class LogNegative(str, Enum):
    """Base for Numpy negated"""

    TRUE = "True"
    FALSE = "False"


@dataclass
class ModelDataTransform(DataTransform):
    """Data transformer that applies and reverses logarithmic functions to user data"""

    @apischema.type_name("ModelDataTransformParams")
    @dataclass
    class Parameters:
        base: Annotated[
            LogBase,
            schema(
                title="Base",
                description="The log, log2 or log10 base to use in log transformation",
            ),
        ] = field(
            default=None,
        )
        negation: Annotated[
            LogNegative,
            schema(
                title="Negation",
                description="Whether or not to make the log transform performed negated (-)",
            ),
        ] = field(
            default=None,
        )
        conversion: Annotated[
            Optional[int],
            schema(
                title="Conversion power",
                description="The conversion power applied in the log transformation",
            ),
        ] = field(
            default=None,
        )

    name: Literal["ModelDataTransform"] = "ModelDataTransform"
    parameters: Parameters = Parameters()

    base_dict = {
        LogBase.LOG2: np.log2,
        LogBase.LOG10: np.log10,
        LogBase.LOG: np.log,
    }

    base_negation = {
        LogNegative.TRUE: True,
        LogNegative.FALSE: False,
    }

    reverse_dict = {
        LogBase.LOG2: lambda x: 2**x,
        LogBase.LOG10: lambda x: 10**x,
        LogBase.LOG: np.exp,
    }

    def transform_df(self, df: pd.Series) -> pd.Series:
        return self.base_dict[self.parameters.base](df)

    def transform_one(self, value: float) -> np.float64:
        return self.base_dict[self.parameters.base](value)

    def reverse_transform_df(self, df: pd.Series) -> pd.Series:
        return self.reverse_dict[self.parameters.base](df)

    def reverse_transform_one(self, value: float) -> np.float64:
        return self.reverse_dict[self.parameters.base](value)

    def transform(self, y_):
        if self.parameters.conversion is not None:
            y_ = y_ / np.power(10, self.parameters.conversion)
        if isinstance(y_, pd.Series):
            transformed = self.transform_df(y_)
        else:
            transformed = self.transform_one(y_)
        if len(transformed.shape) >= 1:
            transformed[~np.isfinite(transformed)] = float("nan")
        if self.base_negation[self.parameters.negation]:
            return -transformed
        else:
            return transformed

    def reverse_transform(self, y_):
        if self.base_negation[self.parameters.negation]:
            y_ = -y_.astype(float)
        if isinstance(y_, pd.Series):
            transformed = self.reverse_transform_df(y_)
        else:
            transformed = self.reverse_transform_one(y_)
        if self.parameters.conversion is not None:
            transformed = transformed * np.power(10, self.parameters.conversion)
        return transformed


class AuxTransformer(DataTransform):
    """Base class for Auxiliary transformation classes

    Each auxiliary data transforation provides the method `transform`,
    which takes raw auxiliary data, and returns numpy arrays with
    transformed auxiliary data."""

    @abc.abstractmethod
    def transform(self, auxiliary_data: np.array) -> np.array:
        pass


@dataclass
class VectorFromColumn(AuxTransformer):
    """Vector from column

    Splits delimited values from in inputs into usable vectors"""

    @apischema.type_name("VectorFromColumnParams")
    @dataclass
    class Parameters:
        delimiter: Annotated[
            str,
            schema(
                title="Delimiter",
                description="String used to split the auxiliary column into a vector",
            ),
        ] = field(
            default=",",
        )

    name: Literal["VectorFromColumn"] = "VectorFromColumn"
    parameters: Parameters = Parameters()

    def transform(self, auxiliary_data: np.array) -> np.array:
        return np.array(
            [
                np.fromstring(val, sep=self.parameters.delimiter)
                for val in auxiliary_data
            ]
        )


@dataclass
class ZScales(AuxTransformer):
    """Z-scales from column

    Calculates Z-scores for sequences or a predefined list of peptide/protein targets"""

    @apischema.type_name("ZScalesParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["ZScales"] = "ZScales"
    parameters: Parameters = Parameters()

    def transform(self, auxiliary_data: np.ndarray) -> np.ndarray:
        from peptides import Peptide

        return np.array([list(Peptide(val).z_scales()) for val in auxiliary_data])


@dataclass
class AmorProt(AuxTransformer):
    """AmorProt from column

    Calculates AmorProt for sequences or a predefined list of peptide/protein targets"""

    @apischema.type_name("AmorProtParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["AmorProt"] = "AmorProt"
    parameters: Parameters = Parameters()

    def __post_init__(self):
        from amorprot import AmorProt

        self.ap = AmorProt()

    def transform(self, auxiliary_data: np.ndarray) -> np.ndarray:
        aux_array = []
        for val_idx, val in enumerate(auxiliary_data):
            try:
                aux_array.append(self.ap.fingerprint(val))
            except KeyError:
                raise DataTransformError(
                    f"AmorProt transform failed on line {val_idx}, for seq: {val}"
                )
        return np.array(aux_array)


AnyAuxTransformer = Union[VectorFromColumn, ZScales, AmorProt]
