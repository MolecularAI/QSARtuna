import abc
import inspect
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import apischema
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
import sklearn
from apischema import deserializer, identity, schema, serialize, serializer
from apischema.conversions import Conversion
from apischema.metadata import skip
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP
from fastprop.descriptors import _descriptor_names_to_mordred_class
from jazzy import logging as jazzy_logging
from jazzy.api import JazzyError, molecular_vector_from_smiles
from joblib import Parallel, delayed, effective_n_jobs
from mordred import Calculator
from numpy import ndarray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from rdkit.DataStructs.cDataStructs import (
    ExplicitBitVect,
    SparseBitVect,
    UIntSparseIntVect,
)
from sklearn import preprocessing

from optunaz.config import NameParameterDataclass
from optunaz.utils import load_df_from_file

jsonpickle_numpy.register_handlers()

logger = logging.getLogger(__name__)
jazzy_logging.logger.setLevel(logging.CRITICAL)
RDLogger.DisableLog("rdApp.*")

RdkitFp = Union[  # Types that RDkit expects as input.
    ExplicitBitVect,  # relatively small (10K bits) or dense bit vectors.
    SparseBitVect,  # large, sparse bit vectors.
    UIntSparseIntVect,  # sparse vectors of integers.
]


class ScalingFittingError(Exception):
    """Raised when insufficient molecules for UnfittedSklearnSclaer to fit"""

    def __init__(self, descriptor_str=None):
        self.descriptor_str = descriptor_str

    pass


class NoValidSmiles(Exception):
    """Raised when no valid SMILES are available"""

    pass


class DescriptorFromConfigError(Exception):
    """Raised when no valid SMILES are available"""

    pass


def mol_from_smi(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smi)
    except TypeError:
        return None

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
    """Molecular Descriptors.

    Descriptors can be fingerprints,
    but can also be custom user-specified descriptors.

    Descriptors calculate a feature vector
    that will be used as input for predictive models
    (e.g. scikit-learn models).
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
        return np.ndarray([])

    def parallel_compute_descriptor(
        self, smiles: List[str], n_cores=None, cache=None
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """Use python Parallel to compute descriptor (e.g. a fingerprint) for a given SMILES string.

        Can be used to generate descriptors in parallel and/or with a cache"""

        def try_cache(mol, cache=cache):
            try:
                return cache.cache(self.calculate_from_smi)(mol)
            except (FileNotFoundError, OSError):  # handle when cleared cache not found
                return self.calculate_from_smi(mol)

        if cache is not None:
            # Scaled and Composite descriptors accept cache in order to sub-cache nested calculate_from_smi
            if "cache" in self.calculate_from_smi.__code__.co_varnames:
                _calculate_from_smi = partial(self.calculate_from_smi, cache=cache)
            # All other descriptors cache calculate_from_smi directly
            else:
                _calculate_from_smi = partial(try_cache, cache=cache)
            if n_cores is None:
                if hasattr(cache, "n_cores"):
                    n_cores = effective_n_jobs(cache.n_cores)
                else:
                    n_cores = effective_n_jobs(-1)
        else:
            _calculate_from_smi = self.calculate_from_smi
            if n_cores is not None:
                n_cores = effective_n_jobs(n_cores)
            else:
                n_cores = effective_n_jobs(-1)
        return Parallel(n_jobs=n_cores)(
            delayed(_calculate_from_smi)(smi) for smi in smiles
        )

    def __post_init__(self):
        try:
            self.__len__ = len(self.calculate_from_smi("C"))
        except TypeError:
            pass


class RdkitDescriptor(MolDescriptor, abc.ABC):
    """Abstract class for RDKit molecular descriptors (fingerprints)."""

    @abc.abstractmethod
    def calculate_from_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Returns a descriptor (fingerprint) for a given RDKit Mol as a 1-d Numpy array."""
        pass

    def calculate_from_smi(self, smi: str) -> Optional[np.ndarray]:
        """Returns a descriptor (fingerprint) for a given SMILES string.

        The descriptor is returned as a 1-d Numpy ndarray.

        Returns None if input SMILES string is not valid according to RDKit.
        """

        mol = mol_from_smi(smi)
        if mol is None:
            return None
        else:
            return self.calculate_from_mol(mol)


@dataclass
class AmorProtDescriptors(MolDescriptor):
    """AmorProtDescriptors

    These descriptors are intended to be used with Peptide SMILES
    """

    try:
        from amorprot import AmorProt

        class _AmorProt(AmorProt):
            """Modified AmorProt class to allow for custom SMILES input."""

            def __init__(
                self,
                maccs=True,
                ecfp4=True,
                ecfp6=True,
                rdkit=True,
                W=10,
                A=10,
                R=0.85,
                smi=None,
            ):
                self.AA_dict = {smi: smi}
                self.maccs = maccs
                self.ecfp4 = ecfp4
                self.ecfp6 = ecfp6
                self.rdkit = rdkit
                self.W = W
                self.A = A
                self.R = R

                if smi is None:
                    return

                try:
                    if self.maccs:
                        self.maccs_dict = {}
                        for aa in self.AA_dict.keys():
                            mol = Chem.MolFromSmiles(self.AA_dict[aa])
                            self.maccs_dict[aa] = np.array(
                                MACCSkeys.GenMACCSKeys(mol)
                            ).tolist()

                    if self.ecfp4:
                        self.ecfp4_dict = {}
                        for aa in self.AA_dict.keys():
                            mol = Chem.MolFromSmiles(self.AA_dict[aa])
                            mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                                radius=2,
                                fpSize=1024,
                            )
                            self.ecfp4_dict[aa] = mfpgen.GetFingerprintAsNumPy(mol)

                    if self.ecfp6:
                        self.ecfp6_dict = {}
                        for aa in self.AA_dict.keys():
                            mol = Chem.MolFromSmiles(self.AA_dict[aa])
                            mfpgen = rdFingerprintGenerator.GetMorganGenerator(
                                radius=3,
                                fpSize=1024,
                            )
                            self.ecfp6_dict[aa] = mfpgen.GetFingerprintAsNumPy(mol)

                    if self.rdkit:
                        self.rdkit_dict = {}
                        for aa in self.AA_dict.keys():
                            mol = Chem.MolFromSmiles(self.AA_dict[aa])
                            mfpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                                fpSize=1024,
                            )
                            self.rdkit_dict[aa] = mfpgen.GetFingerprintAsNumPy(mol)

                except Exception as e:
                    logging.critical(
                        "Failed to initialize AmorProt with SMILES: "
                        f"{smi}. Error: {e}"
                    )
                    raise ValueError

            @abc.abstractmethod
            def calculate_from_smi(self, smi: str) -> np.ndarray:
                try:
                    self.__init__(smi=smi)
                    return self.fingerprint([smi])
                except ValueError:
                    return None

    except ImportError:
        pass

    @apischema.type_name("AmorProtDescParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["AmorProtDescriptors"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str):
        try:
            return self._AmorProt().calculate_from_smi(smi)
        except AttributeError:
            logging.critical(
                "The amorprot package must be installed to use AmorProtDescriptors"
            )
            sys.exit(1)


@dataclass
class Avalon(RdkitDescriptor):
    """Avalon Descriptor

    Avalon (see Gedeck P, et al. QSAR-how good is it in practice?) uses a fingerprint generator in a similar to way
    to Daylight fingerprints, but enumerates with custom feature classes of the molecular graph ( see ref. paper for
    the 16 feature classes used). Hash codes for the path-style features are computed implicitly during enumeration.
    Avalon generated the largest number of good models in the reference study, which is likely since the fingerprint
    generator was tuned toward the features contained in the data set.
    """

    @apischema.type_name("AvalonParams")
    @dataclass
    class Parameters:
        nBits: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        )

    name: Literal["Avalon"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.parameters.nBits)
        return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class AvalonCount(RdkitDescriptor):
    """Avalon Count Descriptor

    AvalonCount (see Gedeck P, et al. QSAR-how good is it in practice?) uses a similar to way to calculate descriptors,
    to the Avalon descriptor, but counts the occurrence of feature classes in the molecular graph ( see ref. paper for
    the 16 feature classes used). Hash codes for the path-style features are computed implicitly during enumeration.
    """

    @apischema.type_name("AvalonCountParams")
    @dataclass
    class Parameters:
        nBits: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="nBits",
                description="Number of bits in the count fingerprint, sometimes also called size.",
            ),
        )

    name: Literal["AvalonCount"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=self.parameters.nBits)
        return numpy_from_rdkit(fp, dtype=int)


@dataclass
class ECFP(RdkitDescriptor):
    """ECFP

    Binary Extended Connectivity Fingerprint (ECFP).\n

    ECFP (see Rogers et al. "Extended-Connectivity Fingerprints.") [also known as Circular Fingerprints or Morgan
    Fingerprints], are built by applying the Morgan algorithm to a set of user-supplied atom invariants. This
    approach (implemented here using GetMorganFingerprintAsBitVect from RDKit) systematically records the
    neighborhood of each non-H atom into multiple circular layers up to a given radius (provided at runtime). The
    substructural features are mapped to integers using a hashing procedure (length of the hash provided at runtime).
    It is the set of the resulting identifiers that defines ECFPs. The diameter of the atom environments is appended
    to the name (e.g. ECFP4 corresponds to radius=2).
    """

    @apischema.type_name("EcfpParams")
    @dataclass
    class Parameters:
        radius: int = field(
            default=3,
            metadata=schema(
                min=1,
                title="radius",
                description="Radius of the atom environments considered."
                " Note that the 4 in ECFP4"
                " corresponds to the diameter of the atom environments considered,"
                " while here we use radius."
                " For example, radius=2 would correspond to ECFP4.",
            ),
        )
        nBits: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        )
        returnRdkit: bool = False

    name: Literal["ECFP"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.parameters.radius,
            fpSize=self.parameters.nBits,
        )

        fp = mfpgen.GetFingerprint(mol)

        if fp is None:
            return None
        if self.parameters.returnRdkit:
            return fp
        else:
            return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class ECFP_counts(RdkitDescriptor):
    """ECFP With Counts

    Binary Extended Connectivity Fingerprint (ECFP) With Counts.\n

    ECFP (see Rogers et al. "Extended-Connectivity Fingerprints.") [also known as Circular Fingerprints or Morgan
    Fingerprints] With Counts are built similar to ECFP fingerprints, however this approach (implemented using
    GetHashedMorganFingerprint from RDKit) systematically records the count vectors rather than bit vectors. Bit
    vectors track whether features appear in a molecule while count vectors track the number of times each
    feature appears. The diameter of the atom environments is appended to the name (e.g. ECFP4 corresponds to radius=2).
    """

    @apischema.type_name("EcfpCountsParams")
    @dataclass
    class Parameters:
        radius: int = field(
            default=3,
            metadata=schema(
                min=1,
                title="radius",
                description="Radius of the atom environments considered. For ECFP4 (diameter=4) set radius=2",
            ),
        )
        useFeatures: bool = field(
            default=True,
            metadata=schema(
                title="useFeatures",
                description="Use feature fingerprints (FCFP),"
                " instead of normal ones (ECFP)."
                " RDKit feature definitions are adapted from the definitions in"
                " Gobbi & Poppinger, Biotechnology and Bioengineering 61, 47-54 (1998)."
                " FCFP and ECFP will likely lead to different fingerprints/similarity scores.",
            ),
        )
        nBits: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        )

    name: Literal["ECFP_counts"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.parameters.radius,
            fpSize=self.parameters.nBits,
        )

        fp = mfpgen.GetCountFingerprintAsNumPy(mol)

        return fp


@dataclass
class PathFP(RdkitDescriptor):
    """PathFP

    Path fingerprint based on RDKit FP Generator.\n

    This is a Path fingerprint.
    """

    @apischema.type_name("PathFPParams")
    @dataclass
    class Parameters:
        maxPath: int = field(
            default=3,
            metadata=schema(
                min=1,
                title="maxPath",
                description="Maximum path for the fingerprint",
            ),
        )
        fpSize: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="fpSize",
                description="Number size of the fingerprint, sometimes also called bit size.",
            ),
        )

    name: Literal["PathFP"]
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            maxPath=self.parameters.maxPath, fpSize=self.parameters.fpSize
        )
        fp = rdkit_gen.GetFingerprintAsNumPy(mol)
        return fp


@dataclass
class MACCS_keys(RdkitDescriptor):
    """MACCS

    Molecular Access System (MACCS) fingerprint.\n

    MACCS fingerprints (often referred to as MDL keys after the developing company ) are calculated using keysets
    originally constructed and optimized for substructure searching (see Durant et al. Reoptimization of MDL keys for
    use in drug discovery) are 166-bit 2D structure fingerprints.\n

    Essentially, they are a binary fingerprint (zeros and ones) that answer 166 fragment related questions. If the
    explicitly defined fragment exists in the structure, the bit in that position is set to 1, and if not,
    it is set to 0. In that sense, the position of the bit matters because it is addressed to a specific question or
    a fragment. An atom can belong to multiple MACCS keys, and since each bit is binary, MACCS 166 keys can represent
    more than 9.3×1049 distinct fingerprint vectors.
    """

    @apischema.type_name("MaccsParams")
    @dataclass
    class Parameters:
        feature_names: Optional[List[str]] = None

    name: Literal["MACCS_keys"]
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        # Get MACCS descriptor names.
        if not self.parameters.feature_names:
            self.parameters.feature_names = [""] + [
                d for _, (d, _) in sorted(MACCSkeys.smartsPatts.items())
            ]
        self.__len__ = len(self.parameters.feature_names)

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        return numpy_from_rdkit(fp, dtype=bool)


@dataclass
class UnfittedSklearnScaler:
    @dataclass
    class MolData:
        file_path: Optional[pathlib.Path] = None
        smiles_column: Optional[str] = None

    mol_data: MolData = field(default_factory=MolData)
    name: Literal["UnfittedSklearnScaler"] = "UnfittedSklearnScaler"

    def get_fitted_scaler_for_fp(self, fp, cache=None):
        scaler = (
            sklearn.preprocessing.StandardScaler()
        )  # Add choice of different scalers.

        df = load_df_from_file(self.mol_data.file_path, self.mol_data.smiles_column)
        try:
            smis = df["canonical"].str.replace(">>", ".")
        except KeyError:
            smis = df[self.mol_data.smiles_column].str.replace(">>", ".")
        fps, failed_idx = descriptor_from_config(smis, fp, cache=cache)

        # handle descriptors that have feature sets, as these types which may
        # contain NaN values in the fingerprints
        nans = []
        if hasattr(fp.parameters, "descriptor_set"):
            nans = np.where(np.isnan(fps).any(axis=0))[0]
            if len(nans) > 0:
                logger.warning(
                    f"Found {len(nans)} NaN descriptor features when scaling {fp}, "
                    "ignoring them in Scaler."
                )
                fps = np.delete(fps, nans, axis=1)
            nans = list(map(int, nans))  # ensure serializable
        if len(failed_idx) > 0:
            logger.warning(
                f"Could not compute descriptors for {len(failed_idx)} SMILES,"
                " ignoring them in Scaler."
            )
        if len(fps) == 0:
            msg = f"{len(fps)} fingerprints too few to train scaler."
            logger.warning(msg)
            raise ScalingFittingError(msg)
        scaler.fit(fps)
        jsonpickle_numpy.register_handlers()  # An extra time.
        saved_params = jsonpickle.dumps(scaler)
        return FittedSklearnScaler(saved_params=saved_params, invalid_idx=nans)


@dataclass
class FittedSklearnScaler:
    saved_params: str
    invalid_idx: Optional[List[int]] = None
    name: Literal["FittedSklearnScaler"] = "FittedSklearnScaler"

    def get_fitted_scaler(self):
        return jsonpickle.loads(self.saved_params)


@dataclass
class UnscaledMAPC(RdkitDescriptor):
    """Unscaled MAPC descriptors

    These MAPC descriptors are unscaled and should be used with caution. MinHashed Atom-Pair Fingerprint Chiral (see
    Orsi et al. One chiral fingerprint to find them all) is the original version of the MinHashed Atom-Pair
    fingerprint of radius 2 (MAP4) which combined circular substructure fingerprints and atom-pair fingerprints into
    a unified framework. This combination allowed for improved substructure perception and performance in small
    molecule benchmarks while retaining information about bond distances for molecular size and shape perception.

    These fingerprints expand the functionality of MAP4 to include encoding of stereochemistry into the fingerprint.
    CIP descriptors of chiral atoms are encoded into the fingerprint at the highest radius. This allows MAPC
    to modulate the impact of stereochemistry on fingerprints, making it scale with increasing molecular size
    without disproportionally affecting structural fingerprints/similarity.
    """

    @apischema.type_name("UnscaledMAPCParams")
    @dataclass
    class Parameters:
        maxRadius: int = field(
            default=2,
            metadata=schema(
                min=1,
                title="maxRadius",
                description="Maximum radius of the fingerprint.",
            ),
        )
        nPermutations: int = field(
            default=2048,
            metadata=schema(
                min=1,
                title="nPermutations",
                description="Number of permutations to perform.",
            ),
        )

    name: Literal["UnscaledMAPC"] = "UnscaledMAPC"
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_mol(self, mol: Chem.Mol):
        try:
            from mapchiral.mapchiral import get_fingerprint
        except ImportError:
            logging.critical("mapchiral must be installed to use MAPC fingerprints")
            sys.exit(1)

        fp = get_fingerprint(
            mol,
            max_radius=self.parameters.maxRadius,
            n_permutations=self.parameters.nPermutations,
        )
        return fp


@dataclass
class UnscaledPhyschemDescriptors(RdkitDescriptor):
    """Base (unscaled) PhyschemDescriptors (RDKit) for PhyschemDescriptors

    These physchem descriptors are unscaled and should be used with caution. They are a set of 208 physchem/molecular
    properties that are calculated in RDKit and used as descriptor vectors for input molecules. Features include
    ClogP, MW, # of atoms, rings, rotatable bonds, fraction sp3 C, graph invariants (Kier indices etc), TPSA,
    Slogp descriptors, counts of some functional groups, VSA  MOE-type descriptors, estimates of atomic charges etc.
    (See https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    Vectors whose components are molecular descriptors have been used as high-level feature representations for
    molecular machine learning. One advantage of molecular descriptor vectors is their interpretability,
    since the meaning of a physicochemical descriptor can be intuitively understood"""

    @apischema.type_name("UnscaledPhyschemDescriptorsParams")
    @dataclass
    class Parameters:
        feature_names: Optional[List[str]] = None

    name: Literal["UnscaledPhyschemDescriptors"] = "UnscaledPhyschemDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        # Get RDKit descriptor names.
        if not self.parameters.feature_names:
            self.parameters.feature_names = [d[0] for d in Descriptors._descList]
        self.__len__ = len(self.parameters.feature_names)

    def calculate_from_mol(self, mol: Chem.Mol):
        d = Descriptors.CalcMolDescriptors(mol)
        if not self.parameters.feature_names:
            d = [d[key] for key in self.parameters.feature_names]
        else:
            d = list(d.values())
        
        if np.isnan(d).any():
            return None
        else:
            return np.array(d)


class JazzyEmbeddingType(str, Enum):
    """embedding_type configures whether the input molecule is embedded in 2D or 3D. Note that 2D embedding is much
    faster and less prone to failures but less accurate (see
    notebooks/gerber_deltag_validation.ipynb)https://jazzy.readthedocs.io/en/latest/cookbook.html#embedding-type
    """

    _2D = "2D"
    "uniform weights. All points in each neighborhood are weighted equally."
    _3D = "3D"
    """weight points by the inverse of their distance so closer neighbors for a query will have greater \
     influence than further neighbors"""


@dataclass
class UnscaledJazzyDescriptors(MolDescriptor):
    """Base (unscaled) Jazzy descriptors

    These Jazzy descriptors are unscaled and should be used with caution. They offer a molecular vector describing
    the hydration free energies and hydrogen-bond acceptor and donor strengths. A publication describing the
    implementation, fitting, and validation of Jazzy can be found at doi.org/10.1038/s41598-023-30089-x. These
    descriptors use the "MMFF94" minimisation method. NB: this descriptor employs a threshold of <50 Hydrogen
    acceptors/donors and a Mw of <1000Da for compound inputs.
    """

    @apischema.type_name("UnscaledJazzyDescriptorsParams")
    @dataclass
    class Parameters:
        feature_names: Optional[List[str]] = field(
            default=None,
            metadata=schema(
                title="Feature Names",
                description="Enables filtering the Jazzy descriptors returned. Defaults to all if not supplied",
            ),
        )
        jazzy_filters: Optional[Dict[str, Any]] = field(
            default=None,
            metadata=schema(
                min=1,
                title="Jazzy Filters",
                description="Filters molecules with high Mw, H-donors or -acceptors which cause Jazzy issues."
                "Note this defaults to NumHAcceptors=25, NumHDonors=25 and MolWt 1600 if not supplied.",
            ),
        )
        embedding_type: Optional[JazzyEmbeddingType] = field(
            default=JazzyEmbeddingType._2D,
            metadata=schema(
                title="Embedding Type",
                description="configures whether the input molecule is embedded in 2D or 3D."
                "Note that 2D embedding is much faster"
                "and less prone to failures but less accurate."
                " (see notebooks/gerber_deltag_validation.ipynb) in the Jazzy repo.",
            ),
        )
        embedding_max_iterations: Optional[int] = field(
            default=25,
            metadata=schema(
                min=1,
                title="Embedding Max Iterations",
                description="Configures max RDKit iterations/attempts to embed the input molecule if 3D embedding is "
                "used. This has no effect if embedding_type is `2D`.",
            ),
        )
        embedding_seed: Optional[int] = field(
            default=42,
            metadata=schema(
                title="Embedding Seed for reproducibility",
                description="This seed is used for a random number generator for the Jazzy embedding for 3D embeddings."
                "This has no effect if embedding_type is `2D`.",
            ),
        )

    name: Literal["UnscaledJazzyDescriptors"] = "UnscaledJazzyDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def _exceeds_descriptor_threshold(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return True
        return any(
            getattr(Descriptors, descriptor)(mol) > threshold
            for descriptor, threshold in self.parameters.jazzy_filters.items()
        )

    def _jazzy_descriptors(self, smi):
        """Returns a Jazzy MMFF94 vector (fingerprint) for a given SMILES string."""
        from jazzy.exception import JazzyError

        if self._exceeds_descriptor_threshold(smi):
            """Raise JazzyError if descriptor input is expected to be slow"""
            raise JazzyError
        """Return the MMFF94 vector"""
        try:
            return molecular_vector_from_smiles(
                smi,
                minimisation_method="MMFF94",
                embedding_type=self.parameters.embedding_type,
                embedding_max_iterations=self.parameters.embedding_max_iterations,
                embedding_seed=self.parameters.embedding_seed,
            )
        except Exception as e:
            logging.warning(
                f"Jazzy failed to calculate descriptors for SMILES '{smi}' with error: {e}."
            )
            return None

    def __post_init__(self):
        # Get Jazzy descriptor names.
        if not self.parameters.feature_names:
            self.parameters.feature_names = sorted(
                list(molecular_vector_from_smiles("C", minimisation_method="MMFF94"))
            )
        if not self.parameters.jazzy_filters:
            self.parameters.jazzy_filters = {
                "NumHAcceptors": 25,
                "NumHDonors": 25,
                "MolWt": 1600,
            }
        self.__len__ = len(self.parameters.feature_names)

    def calculate_from_smi(self, smi: str) -> np.ndarray | None:
        try:
            d = self._jazzy_descriptors(smi)
            d = [d[jazzy_name] for jazzy_name in self.parameters.feature_names]
        except (JazzyError, TypeError, KeyError):
            return None
        if np.isnan(d).any():
            return None
        else:
            return np.array(d)


@dataclass
class UnscaledZScalesDescriptors(MolDescriptor):
    """Unscaled Z-Scales.

    Compute the Z-scales of a peptide SMILES. These Z-Scales descriptors are unscaled and should be used with caution.
    """

    @apischema.type_name("UnscaledZScalesDescParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["UnscaledZScalesDescriptors"] = "UnscaledZScalesDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        from peptides import tables

        self.__len__ = len(tables.Z_SCALES.keys())

    def calculate_from_smi(self, smi: str) -> list[str | list[None]] | None:
        try:
            from chemistry_adapters import AminoAcidAdapter
            from peptides import Peptide
        except ImportError:
            logging.critical(
                "chemistry_adapters and peptides packages must be installed for Z-Scales descriptors"
            )

        try:
            sequence = AminoAcidAdapter().convert_smiles_to_amino_acid_sequence(smi)
            fp = list(Peptide(sequence).z_scales())
        except KeyError:
            return None  # SMILES to aa sequence failed
        return fp


class MordredDescriptorSet(str, Enum):
    """Which version of Mordred to use."""

    ALL_2D = "all"
    "All 2D descriptors as defined by fastprop."
    SUBSET_947 = "optimized"
    "The 947 optimized subset as defined by fastprop."


@dataclass
class MordredDescriptors(RdkitDescriptor):
    """Mordred descriptors.

    Based on the mordred-community descriptor package, these descriptors are a collection of
    over 1800 molecular descriptors, encompassing two- and three-dimensional numerical values that
    precisely represent aspects such as constitutional composition, atom-type electrotopology,
    charge indices, and various facets of molecular geometry and connectivity. A curated subset of
    947 descriptors is also available, which has been optimized for performance and predictivity.

    Mordred descriptors are distinguished from RDKit and basic descriptor sets by offering an
    exceptionally broad and specialized feature set, incorporating a rich suite of topological,
    geometrical, electronic, constitutional, information-theoretic, and autocorrelation descriptors,
    such as Chi, Kappa, and BCUT descriptors, Burden and E-state indices, and various 3D molecular shape
    and connectivity metrics. These descriptors provide insights into molecular complexity, branching,
    information content, symmetry, and fragment diversity not available via other methods. As a result,
    mordred-community enables more nuanced characterization of molecular architecture, stereochemistry,
    and subtle structural motifs, allowing fine-grained features that can be crucial for advanced
    structure-activity relationship studies.
    """

    @apischema.type_name("MordredDescParams")
    @dataclass
    class Parameters:
        descriptor_set: Optional[MordredDescriptorSet] = field(
            default=MordredDescriptorSet.ALL_2D,
            metadata=schema(
                title="Descriptor Set",
                description="Whether to use all or the 947 subset of Mordred descriptors.",
            ),
        )
        feature_names: Optional[List[str]] = field(
            default=None,
            metadata=schema(
                title="Feature Names",
                description="Enables filtering the Mordred descriptors returned. Defaults to all if not supplied",
            ),
        )

    name: Literal["MordredDescriptors"] = "MordredDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def _mordred_descriptors(self):
        # Use the descriptor set directly
        descriptor_set = _descriptor_names_to_mordred_class(
            DESCRIPTOR_SET_LOOKUP.get(self.parameters.descriptor_set)
        )
        if self.parameters.feature_names:
            descriptor_set = [
                d for d in descriptor_set if str(d) in self.parameters.feature_names
            ]
        return Calculator(descriptor_set)

    def __post_init__(self):
        # Get mordred descriptor names.
        if not self.parameters.feature_names:
            # Get all mordred descriptors
            mordred_desc = self._mordred_descriptors()(Chem.MolFromSmiles("C"))
            self.parameters.feature_names = [
                str(desc) for desc in mordred_desc._descriptors
            ]
        self.__len__ = len(self.parameters.feature_names)

    def calculate_from_mol(self, mol: Chem.Mol):
        return np.array(list(self._mordred_descriptors()(mol)), dtype=np.float64)


@dataclass
class PrecomputedDescriptorFromFile(MolDescriptor):
    """Precomputed descriptors.

    Users can supply a CSV file of feature vectors to use as descriptors, with headers on the first line. Each row
    corresponds to a compound in the training set, followed by a column that may have comma-separated vectors describing
    that molecule.
    """

    @apischema.type_name("PrecomputedDescriptorFromFileParams")
    @dataclass
    class Parameters:
        file: Optional[str] = field(
            default=None,
            metadata=schema(
                title="File",
                description="Name of the CSV containing precomputed descriptors",
            ),
        )
        input_column: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Input column",
                description="Name of input column with SMILES strings",
            ),
        )
        response_column: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Response column",
                description="Name of response column with the comma-separated vectors that the model will use as "
                "pre-computed descriptors",
            ),
        )

    name: Literal["PrecomputedDescriptorFromFile"] = "PrecomputedDescriptorFromFile"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        if self.parameters.file == None:
            logger.info(
                "'file' parameter not set. 'inference_parameters' can be used for inference"
            )
            return
        df = pd.read_csv(self.parameters.file, skipinitialspace=True)
        out_cols = [self.parameters.input_column, self.parameters.response_column]
        df = df[out_cols]

        # Add canonicalised SMILES to end of file dataframe
        can_smiles = descriptor_from_config(
            df[self.parameters.input_column],
            CanonicalSmiles.new(),
            return_failed_idx=False,
        )
        can_df = pd.DataFrame({self.parameters.input_column: can_smiles})
        can_df[self.parameters.response_column] = df[self.parameters.response_column]
        df = pd.concat((df, can_df)).reset_index(drop=True)
        df.dropna(subset=out_cols, inplace=True)
        df.drop_duplicates(subset=out_cols, inplace=True)

        # Allow reaction SMILES compatability
        df.loc[:, self.parameters.input_column] = df[
            self.parameters.input_column
        ].str.replace(">>", ".")

        self.df = df[out_cols]

    def calculate_from_smi(self, smi: str) -> np.ndarray:
        rows = self.df[self.df[self.parameters.input_column] == smi]
        if len(rows) < 1:
            logger.warning(
                f"Could not find descriptor for {smi} in file {self.parameters.file}."
            )
            return None
        if len(rows) > 1:
            logger.warning(
                f"Multiple (conflicting) descriptors found for {smi}, taking the first one."
            )
        descriptor_iloc = rows[self.parameters.response_column].iloc[0]
        try:
            return np.array([descriptor_iloc.astype(float)])
        except (ValueError, AttributeError):
            fp = np.fromstring(descriptor_iloc, sep=",")
            if len(fp) == 0:
                return None
            else:
                return fp

    def inference_parameters(self, file: str, input_column: str, response_column: str):
        """This function allows precomputed descriptors to be used for inference for a new file"""

        self.parameters.file = file
        self.parameters.input_column = input_column
        self.parameters.response_column = response_column
        self.__post_init__()


@dataclass
class SmilesFromFile(MolDescriptor):
    """Smiles as descriptors (for ChemProp).

    ChemProp optimization runs require either this or SmilesAndSideInfoFromFile descriptor to be selected.
    This setting allows the SMILES to pass through to the ChemProp package.
    """

    @apischema.type_name("SmilesFromFileParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["SmilesFromFile"] = "SmilesFromFile"

    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str) -> list[str | list[None]] | None:
        # Handle RDkit errors here to avoid poor handling by ChemProp
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        # None is used as placeholder to indicate no side information is used
        return smi


@dataclass
class SmilesAndSideInfoFromFile(MolDescriptor):
    """SMILES & side information descriptors (for ChemProp).

    ChemProp optimization requires either these or SmilesFromFile descriptors. This descriptor allows SMILES to pass
    through to ChemProp, _and_ for side information to be supplied as auxiliary tasks.\n

    Side information can take the form of any vector (continuous or binary) which describe input compounds. All tasks
    are learnt in a multi-task manner to improve main-task (task of intent) predictions. Side information can boost
    performance since their contribution to network loss can lead to improved learnt molecular representations.\n

    Optimal side information weighting (how much auxiliary tasks contribute to network loss) is also
    an (optional) learned parameter during optimization.\n

    Similar to PrecomputedDescriptorFromFile, CSV inputs for this descriptor should contain a SMILES column of
    input molecules. All vectors in the remaining columns are used as user-derived side-information (i.e: be cautious
    to only upload a CSV with side information tasks in columns since _all_ are used)\n

    (see https://ruder.io/multi-task/index.html#auxiliarytasks for details).
    """

    @apischema.type_name("SmilesAndSideInfoFromFileParams")
    @dataclass
    class Parameters:
        @apischema.type_name("SmilesAndSideInfoFromFileAux_Weight_Pc")
        @dataclass
        class Aux_Weight_Pc:
            low: int = field(default=100, metadata=apischema.schema(min=0, max=100))
            high: int = field(default=100, metadata=apischema.schema(min=0, max=100))
            step: int = field(default=20, metadata=apischema.schema(min=1))

        file: Optional[str] = field(
            default=None,
            metadata=schema(
                title="file",
                description="Name of the CSV containing precomputed side-info descriptors",
            ),
        )
        input_column: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Input column",
                description="Name of input column with SMILES strings",
            ),
        )
        y_aux_column: Optional[str] = field(
            default=None,
            metadata=schema(
                title="Y-labels input column",
                description="Name of the input column with y-labels as side information for multi-task learning. "
                "This is optional if you only with to supply x-labels.",
            ),
        )
        y_aux_weight_pc: Aux_Weight_Pc = field(
            default_factory=Aux_Weight_Pc,
            metadata=schema(
                title="Y-label auxiliary weight percentage",
                description="How much (%) auxiliary tasks (side information) contribute (%)"
                "to the loss function optimised during multi-task learning. The larger the number, "
                "the larger the weight of side information during training.",
            ),
        )
        x_aux_column: Optional[str] = field(
            default=None,
            metadata=schema(
                title="X-descriptors input column",
                description="Name of the input column with extra molecular descriptors. "
                "This is optional if you only wish to supply y-labels.",
            ),
        )

    name: Literal["SmilesAndSideInfoFromFile"] = "SmilesAndSideInfoFromFile"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        assert not (
            self.parameters.x_aux_column == None
            and self.parameters.y_aux_column == None
        ), "Must supply x_aux_column and/or y_aux_column"

        if self.parameters.file == None:
            logger.info(
                "'file' parameter not set. 'inference_parameters' can be used for inference"
            )
            return

        self.y_aux = None
        self.x_aux = None

        out_cols = [
            col
            for col in [
                self.parameters.input_column,
                self.parameters.y_aux_column,
                self.parameters.x_aux_column,
            ]
            if col is not None
        ]
        df = pd.read_csv(self.parameters.file, skipinitialspace=True, usecols=out_cols)

        # Add canonicalised SMILES to end of file dataframe
        can_smiles = descriptor_from_config(
            df[self.parameters.input_column],
            CanonicalSmiles.new(),
            return_failed_idx=False,
        )
        can_df = df.copy()
        can_df[self.parameters.input_column] = can_smiles
        df = pd.concat((df, can_df)).reset_index(drop=True)
        df.dropna(subset=self.parameters.input_column, inplace=True)
        df.drop_duplicates(subset=out_cols, inplace=True)

        # Allow reaction SMILES compatability
        df.loc[:, self.parameters.input_column] = df[
            self.parameters.input_column
        ].str.replace(">>", ".")

        from pandas.core.dtypes.common import is_string_dtype

        if self.parameters.y_aux_column:
            self.y_aux = pd.concat(
                (
                    df[self.parameters.input_column],
                    df[self.parameters.y_aux_column].str.split(",", expand=True)
                    if is_string_dtype(df[self.parameters.y_aux_column])
                    else df[self.parameters.y_aux_column],
                ),
                axis=1,
            )
        if self.parameters.x_aux_column:
            self.x_aux = pd.concat(
                (
                    df[self.parameters.input_column],
                    df[self.parameters.x_aux_column].str.split(",", expand=True)
                    if is_string_dtype(df[self.parameters.x_aux_column])
                    else df[self.parameters.x_aux_column],
                ),
                axis=1,
            )

    def calculate_from_smi(self, smi: str) -> list[str, Any] | None:
        # Handle RDkit errors here to avoid poor handling by ChemProp
        mol = mol_from_smi(smi)
        ret = [smi, [None], [None]]
        if mol is None:
            return None
        for aux_idx, aux_df in enumerate([self.y_aux, self.x_aux]):
            if aux_df is not None:
                rows = aux_df[aux_df[self.parameters.input_column] == smi]
                if len(rows) < 1:
                    logger.warning(
                        f"Could not find descriptor for {smi} in file {self.parameters.file}."
                    )
                    empty = np.zeros(len(aux_df.columns) - 1, dtype=np.float32)
                    empty[:] = None
                    ret[aux_idx + 1] = empty.reshape(1, len(empty))
                if len(rows) > 1:
                    logger.warning(
                        f"Multiple (non-distinct) descriptors found for {smi}, taking the first one."
                    )
                descriptor = rows.drop(self.parameters.input_column, axis=1).values[:1]
                ret[aux_idx + 1] = descriptor
        return ret

    def inference_parameters(self, file: str, input_column: str, response_column: str):
        """This function allows SmilesAndSideInfoFromFile to be used for inference for new x_aux in a file"""

        self.parameters.file = file
        self.parameters.input_column = input_column
        self.parameters.x_aux_column = response_column
        self.__post_init__()


SmilesBasedDescriptor = Union[SmilesFromFile, SmilesAndSideInfoFromFile]

AnyUnscaledDescriptor = Union[
    Avalon,
    AvalonCount,
    ECFP,
    ECFP_counts,
    PathFP,
    AmorProtDescriptors,
    MACCS_keys,
    PrecomputedDescriptorFromFile,
    UnscaledMAPC,
    UnscaledPhyschemDescriptors,
    UnscaledJazzyDescriptors,
    UnscaledZScalesDescriptors,
    MordredDescriptors,
]


@dataclass
class ScaledDescriptor(MolDescriptor):
    """Scaled Descriptor.

    This descriptor is not a complete descriptor,
    but instead it wraps and scales another descriptor.\n

    Some algorithms require input to be within certain range, e.g. [-1..1].
    Some descriptors have different ranges for different columns/features.
    This descriptor wraps another descriptor and provides scaled values.
    """

    @dataclass
    class ScaledDescriptorParameters:
        descriptor: AnyUnscaledDescriptor = field(
            default_factory=UnscaledPhyschemDescriptors
        )
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )

    parameters: ScaledDescriptorParameters = field(
        default_factory=ScaledDescriptorParameters
    )
    name: Literal["ScaledDescriptor"] = "ScaledDescriptor"

    _scaler: Any = field(
        default=None, init=False, metadata=skip
    )  # This is (scikit-learn) object with method transform().

    def set_unfitted_scaler_data(
        self, file_path: str, smiles_column: str, cache=None
    ) -> None:
        if isinstance(self.parameters.scaler, UnfittedSklearnScaler):
            self.parameters.scaler.mol_data.file_path = file_path
            self.parameters.scaler.mol_data.smiles_column = smiles_column
        else:
            raise TypeError(
                f"Called 'set_unfitted_scaler_data'"
                f" for scaler of type {type(self.parameters.scaler)}."
            )

        if os.path.exists(self.parameters.scaler.mol_data.file_path):
            try:
                self._ensure_scaler_is_fitted(cache=cache)
            except ScalingFittingError:
                # Allow ScalingFittingError to continue to optimization and prune
                # descriptors with scaling issues later
                pass
        else:
            logger.warning(
                f"Scaler data file is missing:"
                f" {self.parameters.scaler.mol_data.file_path}"
            )

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
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                # Allow ScalingFittingError to continue to optimization and prune
                # descriptors with scaling issues later
                pass

    def _ensure_scaler_is_fitted(self, cache=None):
        if isinstance(self.parameters.scaler, UnfittedSklearnScaler):
            self.parameters.scaler = (
                self.parameters.scaler.get_fitted_scaler_for_fp(
                    self.parameters.descriptor, cache=cache
                )
            )

        # Cache loaded sklearn scaler.
        if self._scaler is None:
            self._scaler = self.parameters.scaler.get_fitted_scaler()

            # Reduce the length of the fitted descriptor if there are invalid feature indices.
            if self.parameters.scaler.invalid_idx:
                self.__len__ -= len(self.parameters.scaler.invalid_idx)
                # Descriptors with explicit feature_names can be filtered
                if hasattr(self.parameters, "feature_names"):
                    valid_descriptors = [
                        name
                        for f_idx, name in enumerate(self.parameters.feature_names)
                        if f_idx not in self.parameters.scaler.invalid_idx
                    ]
                    self.parameters.feature_names = valid_descriptors
                    self.parameters.descriptor.parameters.feature_names = (
                        valid_descriptors
                    )
                    self.parameters.scaler.invalid_idx = []

    def calculate_from_smi(self, smi: str, cache=None) -> Optional[np.ndarray]:
        self._ensure_scaler_is_fitted()

        if cache is not None:
            cache_desc = cache.cache(self.parameters.descriptor.calculate_from_smi)
            desc = cache_desc(smi)
        else:
            desc = self.parameters.descriptor.calculate_from_smi(smi)  # 1d array.

        if desc is None:
            return None

        # Deal with NaN values.
        if self.parameters.scaler.invalid_idx:
            desc = np.delete(desc, np.array(self.parameters.scaler.invalid_idx), axis=0)

        # Scikit-learn scaler takes 2d array, one sample per row. Reshape.
        desc_2d = np.array(desc).reshape(1, -1)  # single sample = single row.

        scaled = self._scaler.transform(desc_2d)

        scaled_1d = scaled[0, :]  # return as 1d array.

        # Clip the values to the range of float32
        min_float32 = np.finfo(np.float32).min
        max_float32 = np.finfo(np.float32).max
        scaled_1d = np.clip(scaled_1d, min_float32, max_float32)

        return scaled_1d


@dataclass
class PhyschemDescriptors(ScaledDescriptor):
    """PhyschemDescriptors (scaled) calculated in RDKit

    A set of 208 physchem/molecular properties that are calculated in RDKit and used as descriptor vectors for input
    molecules. Features include ClogP, MW, # of atoms, rings, rotatable bonds, fraction sp3 C, graph invariants (Kier
    indices etc), TPSA, Slogp descriptors, counts of some functional groups, VSA  MOE-type descriptors, estimates of
    atomic charges etc. (See https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    Vectors whose components are molecular descriptors have been used as high-level feature representations for
    molecular machine learning. One advantage of molecular descriptor vectors is their interpretability,
    since the meaning of a physicochemical descriptor can be intuitively understood
    """

    @apischema.type_name("PhyschemDescParams")
    @dataclass
    class Parameters:
        feature_names: Optional[List[str]] = None
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )
        descriptor: AnyUnscaledDescriptor = field(
            default_factory=UnscaledPhyschemDescriptors
        )

    parameters: Parameters = field(default_factory=Parameters)
    name: Literal["PhyschemDescriptors"] = "PhyschemDescriptors"

    def __post_init__(self):
        # Get RDKit descriptor names.
        if not self.parameters.feature_names:
            self.parameters.feature_names = [d[0] for d in Descriptors._descList]
        self.parameters.descriptor = UnscaledPhyschemDescriptors.new(
            feature_names=self.parameters.feature_names
        )
        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("PhyschemDescriptors scaling failed")
        self.__len__ = len(self.parameters.feature_names)


@dataclass
class JazzyDescriptors(ScaledDescriptor):
    """Scaled Jazzy descriptors

    Jazzy descriptors offer a molecular vector describing the hydration free energies and hydrogen-bond
    acceptor and donor strengths. A publication describing the implementation, fitting, and validation of Jazzy can
    be found at doi.org/10.1038/s41598-023-30089-x. These descriptors use the "MMFF94" minimisation method.
    NB: Jazzy employs a threshold of <50 Hydrogen acceptors/donors and Mw of <1000Da for input compounds.
    """

    @apischema.type_name("JazzyDescParams")
    @dataclass
    class Parameters:
        feature_names: Optional[List[str]] = field(
            default=None,
            metadata=schema(
                title="Feature Names",
                description="Enables filtering the Jazzy descriptors returned. Defaults to all if not supplied",
            ),
        )
        jazzy_filters: Optional[Dict[str, Any]] = field(
            default=None,
            metadata=schema(
                min=1,
                title="Jazzy Filters",
                description="Filters molecules with high Mw, H-donors or -acceptors which cause Jazzy issues."
                "Note this defaults to NumHAcceptors=25, NumHDonors=25 and MolWt 1600 if not supplied.",
            ),
        )
        embedding_type: Optional[JazzyEmbeddingType] = field(
            default=JazzyEmbeddingType._2D,
            metadata=schema(
                title="Embedding Type",
                description="configures whether the input molecule is embedded in 2D or 3D."
                "Note that 2D embedding is much faster"
                "and less prone to failures but less accurate."
                " (see notebooks/gerber_deltag_validation.ipynb) in the Jazzy repo.",
            ),
        )
        embedding_max_iterations: Optional[int] = field(
            default=25,
            metadata=schema(
                min=1,
                title="Embedding Max Iterations",
                description="Configures max RDKit iterations/attempts to embed the input molecule if 3D embedding is "
                "used. This has no effect if embedding_type is `2D`.",
            ),
        )
        embedding_seed: Optional[int] = field(
            default=42,
            metadata=schema(
                title="Embedding seed for reproducibility",
                description="This seed is used for a random number generator for the Jazzy embedding.",
            ),
        )
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )
        descriptor: AnyUnscaledDescriptor = field(
            default_factory=UnscaledJazzyDescriptors
        )

    name: Literal["JazzyDescriptors"] = "JazzyDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        # Get Jazzy descriptor names.
        if not self.parameters.feature_names:
            self.parameters.feature_names = sorted(
                list(molecular_vector_from_smiles("C", minimisation_method="MMFF94"))
            )

        if self.parameters.jazzy_filters is None:
            self.parameters.jazzy_filters = {
                "NumHAcceptors": 25,
                "NumHDonors": 25,
                "MolWt": 1600,
            }

        self.parameters.descriptor = UnscaledJazzyDescriptors.new(
            feature_names=self.parameters.feature_names,
            jazzy_filters=self.parameters.jazzy_filters,
            embedding_type=self.parameters.embedding_type,
            embedding_max_iterations=self.parameters.embedding_max_iterations,
            embedding_seed=self.parameters.embedding_seed,
        )

        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("JazzyDescriptors scaling failed")
        self.__len__ = len(self.parameters.feature_names)


@dataclass
class ScaledMordredDescriptors(ScaledDescriptor):
    """Scaled Mordred descriptors

    Based on the mordred-community descriptor package, these descriptors are a collection of
    over 1800 molecular descriptors, encompassing two- and three-dimensional numerical values that
    precisely represent aspects such as constitutional composition, atom-type electrotopology,
    charge indices, and various facets of molecular geometry and connectivity. A curated subset of
    947 descriptors is also available, which has been optimized for performance and predictivity.

    Mordred descriptors offer a broad and specialized feature set, incorporating a rich suite of topological,
    geometrical, electronic, constitutional, information-theoretic, and autocorrelation descriptors,
    such as Chi, Kappa, and BCUT descriptors, Burden and E-state indices, and various shape
    and connectivity metrics. These descriptors provide insights into molecular complexity, branching,
    information content, symmetry, and fragment diversity not available via other methods. As a result,
    mordred-community enables more nuanced characterization of molecular architecture, stereochemistry,
    and subtle structural motifs, allowing fine-grained features that can be crucial for advanced
    structure-activity relationship studies.

    Certain descriptors return NaN values, which will be removed from the final descriptor vector.
    """

    @apischema.type_name("DescParams")
    @dataclass
    class Parameters:
        descriptor_set: Optional[MordredDescriptorSet] = field(
            default=MordredDescriptorSet.ALL_2D,
            metadata=schema(
                title="Descriptor Set",
                description="Whether to use all or the 947 subset of Mordred descriptors. "
                "Note that any nan descriptors will be removed ",
            ),
        )
        feature_names: Optional[List[str]] = None
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )
        descriptor: AnyUnscaledDescriptor = field(default_factory=MordredDescriptors)

    parameters: Parameters = field(default_factory=Parameters)
    name: Literal["ScaledMordredDescriptors"] = "ScaledMordredDescriptors"

    def __post_init__(self):
        self.parameters.descriptor = MordredDescriptors.new(
            descriptor_set=self.parameters.descriptor_set,
            feature_names=self.parameters.feature_names,
        )
        # Get initial mordred descriptor names (we will filter for NaN later).
        if not self.parameters.feature_names:
            # Get all mordred descriptors
            mordred_desc = self.parameters.descriptor._mordred_descriptors()(
                Chem.MolFromSmiles("C")
            )
            self.parameters.feature_names = [
                str(desc) for desc in mordred_desc._descriptors
            ]
        self.__len__ = len(self.parameters.feature_names)
        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("Mordred descriptor scaling failed")


@dataclass
class MAPC(ScaledDescriptor):
    """Scaled MAPC descriptors

    MAPC (MinHashed Atom-Pair Fingerprint Chiral) (see Orsi et al. One chiral fingerprint to find them all) is the
    original version of the MinHashed Atom-Pair fingerprint of radius 2 (MAP4) which combined circular substructure
    fingerprints and atom-pair fingerprints into a unified framework. This combination allowed for improved
    substructure perception and performance in small molecule benchmarks while retaining information about bond
    distances for molecular size and shape perception.

    These fingerprints expand the functionality of MAP4 to include encoding of stereochemistry into the fingerprint.
    CIP descriptors of chiral atoms are encoded into the fingerprint at the highest radius. This allows MAPC
    to modulate the impact of stereochemistry on fingerprints, making it scale with increasing molecular size
    without disproportionally affecting structural fingerprints/similarity.
    """

    @apischema.type_name("MAP4CParams")
    @dataclass
    class Parameters:
        maxRadius: int = 2
        nPermutations: int = 2048
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )
        descriptor: AnyUnscaledDescriptor = field(default_factory=UnscaledMAPC)

    name: Literal["MAPC"] = "MAPC"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        self.parameters.descriptor = UnscaledMAPC.new(
            maxRadius=self.parameters.maxRadius,
            nPermutations=self.parameters.nPermutations,
        )

        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("MAPC scaling failed")
        self.__len__ = len(self.parameters.descriptor.calculate_from_smi("C"))


@dataclass
class ZScalesDescriptors(ScaledDescriptor):
    """Scaled Z-Scales descriptors.

    Z-scales were proposed in Sandberg et al (1998) based on physicochemical properties of proteogenic and
    non-proteogenic amino acids, including NMR data and thin-layer chromatography (TLC) data. Refer to
    doi:10.1021/jm9700575 for the original publication. These descriptors capture 1. lipophilicity, 2. steric
    properties (steric bulk and polarizability), 3. electronic properties (polarity and charge),
    4. electronegativity (heat of formation, electrophilicity and hardness) and 5. another electronegativity.
    This fingerprint is the computed average of Z-scales of all the amino acids in the peptide.
    """

    @apischema.type_name("ZScalesDescParams")
    @dataclass
    class Parameters:
        scaler: Union[FittedSklearnScaler, UnfittedSklearnScaler] = field(
            default_factory=UnfittedSklearnScaler
        )
        descriptor: AnyUnscaledDescriptor = field(
            default_factory=UnscaledZScalesDescriptors
        )

    name: Literal["ZScalesDescriptors"] = "ZScalesDescriptors"
    parameters: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        from peptides import tables

        self.parameters.descriptor = UnscaledZScalesDescriptors.new()

        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("ZScales scaling failed")
        self.__len__ = len(tables.Z_SCALES.keys())


CompositeCompatibleDescriptor = Union[
    AnyUnscaledDescriptor,
    ScaledDescriptor,
    MAPC,
    PhyschemDescriptors,
    JazzyDescriptors,
    ZScalesDescriptors,
    MordredDescriptors,
    ScaledMordredDescriptors,
]


@dataclass
class CompositeDescriptor(MolDescriptor):
    """Composite descriptor

    Concatenates multiple descriptors into one. Select multiple algorithms from the button below. Please note the
    ChemProp SMILES descriptors are not compatible with this function.
    """

    @apischema.type_name("CompositeDescParams")
    @dataclass
    class Parameters:
        descriptors: List[CompositeCompatibleDescriptor]

    parameters: Parameters = field(
        default_factory=lambda: CompositeDescriptor.Parameters(descriptors=[])
    )
    name: Literal["CompositeDescriptor"] = "CompositeDescriptor"

    def calculate_from_smi(self, smi: str, cache=None) -> Optional[np.ndarray]:
        if cache is not None:
            ds = []
            for d in self.parameters.descriptors:
                try:
                    # Composite descriptors comprising scaled descriptors pass through cache
                    ds.append(d.calculate_from_smi(smi, cache=cache))
                except TypeError:
                    # All other descriptors use the cache directly and do not expect cache as parameter
                    sub_calculate_from_smi = cache.cache(d.calculate_from_smi)
                    ds.append(sub_calculate_from_smi(smi))
        else:
            ds = [d.calculate_from_smi(smi) for d in self.parameters.descriptors]
        if any(d is None for d in ds):
            # If any of the descriptors is None,
            # we declare the whole composite descriptor to be None.
            return None
        else:
            concatenated = np.concatenate(ds)
            return concatenated

    def fp_info(self):
        return {
            json.dumps(serialize(d)): len(d.calculate_from_smi("C"))
            for d in self.parameters.descriptors
        }


AnyChemPropIncompatible = Union[CompositeCompatibleDescriptor, CompositeDescriptor]

FeaturizerDescriptors = Union[
    Avalon,
    AvalonCount,
    MACCS_keys,
    UnscaledJazzyDescriptors,
    UnscaledPhyschemDescriptors,
    UnscaledMAPC,
    MordredDescriptors,
]

AnyDescriptor = Union[AnyChemPropIncompatible, SmilesBasedDescriptor]


@dataclass
class CanonicalSmiles(MolDescriptor):
    """Canonical Smiles for use in utility functions (not for user selection)."""

    @apischema.type_name("CanonicalSmilesParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["CanonicalSmiles"] = "CanonicalSmiles"
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str) -> Any | None:
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


@dataclass
class Scaffold(MolDescriptor):
    """Scaffold Smiles for use in utility functions (not for user selection)."""

    @apischema.type_name("ScaffoldParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["Scaffold"] = "Scaffold"
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str) -> Any | None:
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(GetScaffoldForMol((mol)))


@dataclass
class GenericScaffold(MolDescriptor):
    """Generic Scaffold Smiles for use in utility functions (not for user selection)."""

    @apischema.type_name("GenericScaffoldParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["GenericScaffold"] = "GenericScaffold"
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str) -> Any | None:
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        else:
            return Chem.MolToSmiles(MakeScaffoldGeneric(GetScaffoldForMol((mol))))


@dataclass
class ValidDescriptor(MolDescriptor):
    """Validates Smiles for use in utility functions (not for user selection)."""

    @apischema.type_name("ValidDescriptorParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["ValidDescriptor"] = "ValidDescriptor"
    parameters: Parameters = field(default_factory=Parameters)

    def calculate_from_smi(self, smi: str) -> bool:
        """Returns a descriptor (fingerprint) for a given SMILES string.

        The descriptor is returned as a 1-d Numpy ndarray.

        Returns None if input SMILES string is not valid according to RDKit.
        """

        mol = mol_from_smi(smi)
        if mol is None:
            return False
        else:
            return True

    def __post_init__(self):
        # Do not perform the post init from MolDescriptor
        pass


def descriptor_from_config(
    smiles: List[str], descriptor: AnyDescriptor, cache=None, return_failed_idx=True
) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Returns molecular descriptors (fingerprints) for a given set of SMILES and configuration.

    When return_failed_idx is True, this returns a 2d numpy array and valid indices for that descriptor
    When return_failed_idx is False, this returns the raw descriptor output (e.g. for canonical smiles etc)
    """

    if smiles is None or len(smiles) < 1:
        raise NoValidSmiles(
            f"Descriptor {descriptor} cannot generate empty smiles: {smiles}"
        )
    descriptors = descriptor.parallel_compute_descriptor(smiles, cache=cache)
    if return_failed_idx:
        list_of_arrays = []
        failed_idx = []
        for d_idx, d in enumerate(descriptors):
            if d is None or (
                not isinstance(descriptor, (SmilesBasedDescriptor, MordredDescriptors))
                and pd.isnull(np.array(d)).any()
            ):
                failed_idx.append(d_idx)
            else:
                list_of_arrays.append(d)
        if len(list_of_arrays) > 0:
            try:
                list_of_arrays = np.stack(list_of_arrays, axis=0)
            except ValueError:
                list_of_arrays = np.stack(
                    np.array(list_of_arrays, dtype=object), axis=0
                )
        return list_of_arrays, failed_idx
    else:
        return descriptors


def combine_covariates(
    descriptor: AnyDescriptor, X: ndarray, train_aux: ndarray
) -> np.ndarray:
    # we need to make sure that smiles based descriptors append the covariates so that they are in X[:,2] (and to create X[:,1] as an empty column if necessary)
    if isinstance(descriptor, SmilesFromFile):
        # If the descriptor is a Simple SMILES based descriptor, we need to add an empty column at X[:,1] for y_aux
        # This is because ChemProp expects shape (X) == (n_samples, 3)
        return np.array(
            [[x, [None], [train_aux[row]]] for row, x in enumerate(X)], dtype=object
        )
    elif isinstance(descriptor, SmilesAndSideInfoFromFile):
        # SmilesAndSideInfoFromFile only needs to append the covariates to X[:,2] (if they are not initialised to None)
        # If not x_aux infomration is given then we take the covariates.
        return np.array(
            [
                [
                    x[0],
                    x[1],
                    [train_aux[row]]
                    if x[2][0] is None or (x[2][0] == None).all()
                    else [np.hstack((x[2][0], train_aux[row]))],
                ]
                for row, x in enumerate(X)
            ],
            dtype=object,
        )
    elif isinstance(descriptor, AnyChemPropIncompatible):
        # If the descriptor is AnyChemPropIncompatible, we simply append the covariates to X
        if train_aux.ndim == 1:
            train_aux = train_aux.reshape(-1, 1)
        X = np.hstack((X, train_aux), dtype=object)
    else:
        raise TypeError(
            f"Unsupported descriptor type for covariates combination: {type(descriptor)}"
        )
    return X
