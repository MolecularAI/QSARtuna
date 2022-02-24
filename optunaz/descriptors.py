import abc
import inspect
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Union, Type, Optional, Any

import apischema
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
import sklearn
from apischema import deserializer, serializer, schema
from apischema.conversions import Conversion, identity
from apischema.metadata import skip
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.DataStructs.cDataStructs import (
    ExplicitBitVect,
    SparseBitVect,
    UIntSparseIntVect,
)
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn import preprocessing
from typing_extensions import Literal

from optunaz.config import NameParameterDataclass

jsonpickle_numpy.register_handlers()

logger = logging.getLogger(__name__)


RdkitFp = Union[  # Types that RDkit expects as input.
    ExplicitBitVect,  # relatively small (10K bits) or dense bit vectors.
    SparseBitVect,  # large, sparse bit vectors.
    UIntSparseIntVect,  # sparse vectors of integers.
]


def mol_from_smi(smi: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        logger.warning(f"Failed to parse SMILES {smi}.")
        return None

    err_code = Chem.SanitizeMol(mol, catchErrors=True)

    if err_code != 0:
        logger.warning(f"Failed to sanitize SMILES {smi}.")
        return None

    return mol


def numpy_from_rdkit(fp: RdkitFp, dtype: Type) -> np.ndarray:
    """Returns Numpy representation of a given RDKit Fingerprint."""
    fp_numpy = np.zeros((0,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp, fp_numpy)
    return fp_numpy


class MolDescriptor(NameParameterDataclass, abc.ABC):
    """Abstract class for Molecular Descriptors.

    Descriptors can be fingerprints,
    but can also be custom user-specified descriptors.

    Descriptors calculate a feature vector
    that will be used as input for predictive models
    (e.g. scikit-learn models).

    During Optuna hyperparameter optimization,
    descriptors are converted to and from strings.
    Conversion is done by Pydantic,
    that's why MolDescriptors derive from pydantic.BaseModel.
    """

    _union: Any = None

    # You can use __init_subclass__ to register new subclass automatically
    def __init_subclass__(cls, **kwargs):
        if inspect.isabstract(cls):
            return  # Do not register abstract classes, like RDKitDescriptor.
        # Deserializers stack directly as a Union
        deserializer(Conversion(identity, source=cls, target=MolDescriptor))
        # Only Base serializer must be registered (and updated for each subclass) as
        # a Union, and not be inherited
        MolDescriptor._union = (
            cls if MolDescriptor._union is None else Union[MolDescriptor._union, cls]
        )
        serializer(
            Conversion(
                identity,
                source=MolDescriptor,
                target=MolDescriptor._union,
                inherited=False,
            )
        )

    @abc.abstractmethod
    def calculate_from_smi(self, smi: str) -> np.ndarray:
        """Returns a descriptor (e.g. a fingerprint) for a given SMILES string.

        The descriptor is returned as a 1-d Numpy ndarray.
        """
        pass


class RdkitDescriptor(MolDescriptor, abc.ABC):
    """Abstract class for RDKit molecular descriptors (fingerprints)."""

    @abc.abstractmethod
    def calculate_from_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Returns a descriptor (fingerprint) for a given RDKit Mol as a 1-d Numpy array."""
        pass

    def calculate_from_smi(self, smi: str) -> np.ndarray:
        """Returns a descriptor (fingerprint) for a given SMILES string.

        The descriptor is returned as a 1-d Numpy ndarray.
        """

        mol = mol_from_smi(smi)
        return self.calculate_from_mol(mol)


@dataclass
class Avalon(RdkitDescriptor):
    """Avalon Descriptor.

    This descriptor uses RDKit wrapper for Avalon Toolkit http://sourceforge.net/projects/avalontoolkit/

    Reference paper:
    * Gedeck P, Rohde B, Bartels C. QSAR - how good is it in practice?
        Comparison of descriptor sets on an unbiased cross section of corporate data sets.
        J Chem Inf Model. 2006;46(5):1924-1936. doi:10.1021/ci050413p
    """

    @apischema.type_name("AvalonParams")
    @dataclass
    class Parameters:
        nBits: int = field(default=2048, metadata=schema(min=1))

    name: Literal["Avalon"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.parameters.nBits)
        return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class ECFP(RdkitDescriptor):
    """ECFP - Extended Connectivity Fingerprint (MorganFingerprint from RDKit).

    From RDkit documentation:

    This family of fingerprints, also known as circular fingerprints, also known as
    ECFP (see Rogers and Hahn "Extended-Connectivity Fingerprints." J. Chem. Inf. and
    Model. 50:742-54, 2010), is built by applying the Morgan algorithm to a set of
    user-supplied atom invariants. When generating Morgan fingerprints, the radius of
    the fingerprint must also be provided.

    When comparing the ECFP/FCFP fingerprints and the Morgan fingerprints generated
    by the RDKit, remember that the 4 in ECFP4 corresponds to the diameter of the
    atom environments considered, while the Morgan fingerprints take a radius
    parameter. So the examples above, with radius=2, are roughly equivalent to ECFP4
    and FCFP4.
    """

    @apischema.type_name("EcfpParams")
    @dataclass
    class Parameters:
        radius: int = field(
            default=3, metadata=schema(min=1)
        )  #: Radius of the atom environments considered.
        nBits: int = field(
            default=2048, metadata=schema(min=1)
        )  #: Number of bits in the fingerprint, sometimes also called "size".

    name: Literal["ECFP"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.parameters.radius, self.parameters.nBits
        )
        return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class ECFP_counts(RdkitDescriptor):
    """ECFP with counts - Extended Connectivity Fingerprint (MorganFingerprint from RDKit)."""

    @apischema.type_name("EcfpCountsParams")
    @dataclass
    class Parameters:
        radius: int = field(default=3, metadata=schema(min=1))
        useFeatures: bool = True
        nBits: int = field(default=2048, metadata=schema(min=1))

    name: Literal["ECFP_counts"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = AllChem.GetHashedMorganFingerprint(
            mol,
            radius=self.parameters.radius,
            nBits=self.parameters.nBits,
            useFeatures=self.parameters.useFeatures,
        )
        return numpy_from_rdkit(fp, dtype=np.int8)


@dataclass
class MACCS_keys(RdkitDescriptor):
    """MACCS fingerprint."""

    @apischema.type_name("MaccsParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["MACCS_keys"]
    parameters: Parameters = Parameters()

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class UnfittedSklearnScaler:
    @dataclass
    class MolData:
        file_path: pathlib.Path = None
        smiles_column: str = None

    mol_data: MolData = MolData()
    name: Literal["UnfittedSklearnScaler"] = "UnfittedSklearnScaler"

    def get_fitted_scaler_for_fp(self, fp):
        scaler = (
            sklearn.preprocessing.StandardScaler()
        )  # Add choice of different scalers.
        import pandas as pd  # Import pandas here to remove it from inference env.

        df = pd.read_csv(self.mol_data.file_path, skipinitialspace=True)
        smis = df[self.mol_data.smiles_column]
        fps = [fp.calculate_from_smi(smi) for smi in smis]
        scaler.fit(fps)
        jsonpickle_numpy.register_handlers()  # An extra time.
        saved_params = jsonpickle.dumps(scaler)
        return FittedSklearnScaler(saved_params=saved_params)


@dataclass
class FittedSklearnScaler:

    saved_params: str
    name: Literal["FittedSklearnScaler"] = "FittedSklearnScaler"

    def get_fitted_scaler(self):
        return jsonpickle.loads(self.saved_params)


@dataclass
class PhyschemDescriptors(RdkitDescriptor):
    """PhyschemDescriptors - a set of 200 physchem/molecular properties from RDKit.

    This includes ClogP, MW, # of atoms, rings, rotatable bonds, fraction sp3 C,
    graph invariants (Kier indices etc), TPSA, Slogp descriptors, counts of some
    functional groups, VSA  MOE-type descriptors, estimates of atomic charges etc.
    See http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
    You can view the full list with:

        from rdkit.Chem import Descriptors
        print([x[0] for x in Descriptors._descList])

    This descriptor can be combined with ScaledDescriptor.
    """

    @apischema.type_name("PhyschemDescParams")
    @dataclass
    class Parameters:
        rdkit_names: Optional[List[str]] = None

    name: Literal["PhyschemDescriptors"]
    parameters: Parameters = Parameters()

    def _rdkit_descriptors(self):
        return MolecularDescriptorCalculator(self.parameters.rdkit_names)

    def __post_init__(self):

        # Get RDKit descriptor names.
        if self.parameters.rdkit_names is None:
            self.parameters.rdkit_names = [d[0] for d in Descriptors._descList]

    def calculate_from_mol(self, mol: Chem.Mol):
        d = self._rdkit_descriptors().CalcDescriptors(mol)
        return np.array(d)


@dataclass
class PrecomputedDescriptorFromFile(MolDescriptor):
    """Experimental implementation of precomputed descriptors."""

    @apischema.type_name(
        "PrecomputedDescriptorFromFileParams"
    )
    @dataclass
    class Parameters:
        file: str
        input_column: str
        response_column: str

    name: Literal["PrecomputedDescriptorFromFile"]
    parameters: Parameters

    def calculate_from_smi(self, smi: str) -> np.ndarray:
        file = os.environ.get("QPTUNA_PrecomputedDescriptorFromFile_FILENAME", None)
        if file is None:
            file = self.parameters.file
        df = pd.read_csv(file, skipinitialspace=True)
        rows = df[df[self.parameters.input_column] == smi]
        if len(rows) < 1:
            raise ValueError(f"Could not find descriptor for {smi} in file {self.parameters.file}.")
        if len(rows) > 1:
            logger.warning(f"Multiple descriptors found for {smi}, taking the first one.")
        descriptor_str = rows[self.parameters.response_column].iloc[0]
        descriptor = np.fromstring(descriptor_str, sep=",")
        return descriptor


AnyUnscaledDescriptor = Union[
    Avalon,
    ECFP,
    ECFP_counts,
    MACCS_keys,
    PhyschemDescriptors,
    PrecomputedDescriptorFromFile,
]


@dataclass
class ScaledDescriptor(MolDescriptor):
    """Scaled Descriptor.

    This descriptor is not a complete descriptor,
    but instead it wraps and scales another descriptor.

    Some algorithms require input to be within certain range, e.g. [-1..1].
    Some descriptors have different ranges for different columns/features.
    This descriptor wraps another descriptor and provides scaled values.

    Scaler can be specified as UnfittedSklearnScaler or FittedSklearnScaler.
    UnfittedSklearScaler takes a dataset as input,
    calculates underlying descriptor on that dataset,
    and uses that data to fit a scikit-learn Scaler,
    which is then returned as FittedSklearnScaler.
    FittedSklearnScaler is simply a pickled version of a scikit-learn Scaler
    that was fitted to a dataset.

    Most end-users should use UnfittedSklearnScaler.
    Expert users can use FittedSklearnScaler,
    by providing a fitted scaler that conforms to Scikit-learn interface:

    >>> mols = ["CCC", "CCCCC"]                        # Mols for fitting the scaler.
    >>> d = PhyschemDescriptors.new()                  # Unscaled descriptor.
    >>> x = [d.calculate_from_smi(m) for m in mols]    # Values from unscaled version.
    >>> scaler = sklearn.preprocessing.MaxAbsScaler()  # Any scaler here.
    >>> scaler.fit(x)
    >>> d_scaled = ScaledDescriptor.new(
    >>>     descriptor=PhyschemDescriptors.new()
    >>>     scaler=FittedSklearnScaler(
    >>>         saved_params=jsonpickle.dumps(scaler)  # Json-pickled scaler.
    >>>     )
    >>> )

    Any UnfittedSklearnScaler is replaced by FittedSklearnScaler during initialization.
    """

    @dataclass
    class ScaledDescriptorParameters:
        descriptor: AnyUnscaledDescriptor
        scaler: Union[
            FittedSklearnScaler,
            UnfittedSklearnScaler,
        ] = UnfittedSklearnScaler()

    parameters: ScaledDescriptorParameters
    name: Literal["ScaledDescriptor"] = "ScaledDescriptor"

    _scaler: Any = field(
        default=None, init=False, metadata=skip
    )  # This is (scikit-learn) object with method transform().

    def set_unfitted_scaler_data(self, file_path: str, smiles_column: str) -> None:
        if isinstance(self.parameters.scaler, UnfittedSklearnScaler):
            self.parameters.scaler.mol_data.file_path = file_path
            self.parameters.scaler.mol_data.smiles_column = smiles_column
        else:
            raise TypeError(
                f"Called 'set_unfitted_scaler_data'"
                f" for scaler of type {type(self.parameters.scaler)}."
            )

        if os.path.exists(self.parameters.scaler.mol_data.file_path):
            self._ensure_scaler_is_fitted()
        else:
            logger.warning(f"Scaler data file is missing:"
                           f" {self.parameters.scaler.mol_data.file_path}")

    def __post_init__(self):

        # We should avoid heavy computations in init/post_init,
        # but descriptors are serialized+deserialized
        # by optuna objective
        # already before the first use,
        # so we compute the scaler here
        # to make sure we always serialize fitted scaler.
        # Except when path is None,
        # and is provided during `OptimizationConfig.__post_init_()`.
        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            self._ensure_scaler_is_fitted()

    def _ensure_scaler_is_fitted(self):

        if isinstance(self.parameters.scaler, UnfittedSklearnScaler):
            self.parameters.scaler = self.parameters.scaler.get_fitted_scaler_for_fp(
                self.parameters.descriptor
            )

        # Cache unpickled sklearn scaler.
        if self._scaler is None:
            self._scaler = self.parameters.scaler.get_fitted_scaler()

    def calculate_from_smi(self, smi: str) -> np.ndarray:

        self._ensure_scaler_is_fitted()

        desc = self.parameters.descriptor.calculate_from_smi(smi)  # 1d array.

        # Scikit-learn scaler takes 2d array, one sample per row. Reshape.
        desc_2d = np.array(desc).reshape(1, -1)  # single sample = single row.

        scaled = self._scaler.transform(desc_2d)

        scaled_1d = scaled[0, :]  # return as 1d array.

        return scaled_1d


@dataclass
class CompositeDescriptor(MolDescriptor):
    """Composite descriptor: concatenates multiple descriptors into one."""

    @apischema.type_name("CompositeDescParams")
    @dataclass
    class Parameters:
        descriptors: List[MolDescriptor]

    parameters: Parameters
    name: Literal["CompositeDescriptor"] = "CompositeDescriptor"

    def calculate_from_smi(self, smi: str) -> np.ndarray:
        ds = [d.calculate_from_smi(smi) for d in self.parameters.descriptors]
        concatenated = np.concatenate(ds)
        return concatenated


AnyDescriptor = Union[AnyUnscaledDescriptor, ScaledDescriptor, CompositeDescriptor]


def descriptor_from_config(
    smiles: List[str], descriptor: AnyDescriptor
) -> Optional[np.ndarray]:
    """Returns molecular descriptors (fingerprints) for a given set of SMILES and configuration.

    Returns a 2d numpy array.
    Returns None if input `smiles` is None.
    """

    if smiles is None or len(smiles) < 1:
        return None

    list_of_arrays = [descriptor.calculate_from_smi(smi) for smi in smiles]
    array2d = np.stack(list_of_arrays, axis=0)
    return array2d
