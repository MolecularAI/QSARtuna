import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Dict

from apischema import deserialize

from optunaz.utils import mkdict


class ModelMode(str, Enum):
    """Model mode, either regression or classification."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class OptimizationDirection(str, Enum):
    """Optimization direction, either minimization or maximization."""

    MINIMIZATION = "minimize"
    MAXIMIZATION = "maximize"


class Task(str, Enum):
    """Task: optimization, building, or prediction."""

    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    BUILDING = "building"


class NameParameterDataclass(abc.ABC):
    """A base class for (data-)classes that follow "name"-"parameter" structure.

    Here is example of a name-parameter class:

    >>> class ECFP(NameParameterDataclass):
    >>>     name: str
    >>>     parameters: Dict

    This name-parameter structure is used for parsing Json.

    Normally, this abstract class should declare two abstract properties:

    >>> @property
    >>> @abstractmethod
    >>> def name(self) -> str:
    >>>     pass
    >>>
    >>> @property
    >>> @abstractmethod
    >>> def name(self) -> Any:
    >>>     pass

    However, Pydantic does not allow overriding properties,
    thus we don't declare them.
    """

    @classmethod
    def new(cls, **kwargs):
        """Convenience method to initialize objects, instead of __init__.

        For example, the following is a full version that calls __init__:
        >>> descriptor = Avalon(name='Avalon', parameters=Avalon.Parameters(nBits=1024))

        This method allows a shorter version:
        >>> descriptor = Avalon.new(nBits=1024)
        """

        # Dataclasses can't handle recursive instantiation, use apischema.
        # Apischema requires plain dict, can't handle instantiated classes,
        # so convert everything to dict first.
        return deserialize(cls, {"name": cls.__name__, "parameters": mkdict(kwargs)})


class Algorithm(NameParameterDataclass):
    """Abstract class for ML algorithms."""

    pass


@dataclass
class Visualization:
    """Visualization configuration."""

    class ImageFileFormat(str, Enum):
        PNG = "png"
        JPEG = "jpeg"
        JPG = "jpg"
        PDF = "pdf"
        SVG = "svg"

    @dataclass
    class Plots:
        plot_history: bool = False
        plot_contour: bool = False
        plot_parallel_coordinate: bool = False
        plot_slice: bool = False

    output_folder: Optional[Path]
    file_format: Optional[ImageFileFormat]
    plots: Optional[Plots]
    use_xvfb: bool = False
