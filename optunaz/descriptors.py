import abc
import inspect
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Union, Type, Optional, Any, Tuple, Dict
from functools import partial

import apischema
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
import sklearn
from apischema import deserializer, serialize, serializer, schema, identity, type_name
from apischema.conversions import Conversion
from apischema.metadata import skip
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import (
    GetScaffoldForMol,
    MakeScaffoldGeneric,
)
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import (
    ExplicitBitVect,
    SparseBitVect,
    UIntSparseIntVect,
)

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from jazzy.api import molecular_vector_from_smiles
from jazzy.exception import JazzyError
from sklearn import preprocessing
from typing_extensions import Literal, Annotated
from joblib import Parallel, delayed, effective_n_jobs
from optunaz.config import NameParameterDataclass
from optunaz.utils import load_df_from_file

jsonpickle_numpy.register_handlers()

logger = logging.getLogger(__name__)


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
        pass

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
        nBits: Annotated[
            int,
            schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        ] = field(
            default=2048,
        )

    name: Literal["Avalon"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.parameters.nBits)
        return numpy_from_rdkit(fp, dtype=bool)


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
        radius: Annotated[
            int,
            schema(
                min=1,
                title="radius",
                description="Radius of the atom environments considered."
                " Note that the 4 in ECFP4"
                " corresponds to the diameter of the atom environments considered,"
                " while here we use radius."
                " For example, radius=2 would correspond to ECFP4.",
            ),
        ] = field(
            default=3,
        )
        nBits: Annotated[
            int,
            schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        ] = field(
            default=2048,
        )
        returnRdkit: bool = False

    name: Literal["ECFP"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.parameters.radius, self.parameters.nBits
        )
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
        radius: Annotated[
            int,
            schema(
                min=1,
                title="radius",
                description="Radius of the atom environments considered. For ECFP4 (diameter=4) set radius=2",
            ),
        ] = field(default=3)
        useFeatures: Annotated[
            bool,
            schema(
                title="useFeatures",
                description="Use feature fingerprints (FCFP),"
                " instead of normal ones (ECFP)."
                " RDKit feature definitions are adapted from the definitions in"
                " Gobbi & Poppinger, Biotechnology and Bioengineering 61, 47-54 (1998)."
                " FCFP and ECFP will likely lead to different fingerprints/similarity scores.",
            ),
        ] = True
        nBits: Annotated[
            int,
            schema(
                min=1,
                title="nBits",
                description="Number of bits in the fingerprint, sometimes also called size.",
            ),
        ] = field(default=2048)

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
class PathFP(RdkitDescriptor):
    """PathFP

    Path fingerprint based on RDKit FP Generator.\n

    This is a Path fingerprint.
    """

    @apischema.type_name("PathFPParams")
    @dataclass
    class Parameters:
        maxPath: Annotated[
            int,
            schema(
                min=1,
                title="maxPath",
                description="Maximum path for the fingerprint",
            ),
        ] = field(default=3)
        fpSize: Annotated[
            int,
            schema(
                min=1,
                title="fpSize",
                description="Number size of the fingerprint, sometimes also called bit size.",
            ),
        ] = field(default=2048)

    name: Literal["PathFP"]
    parameters: Parameters

    def calculate_from_mol(self, mol: Chem.Mol):
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            maxPath=self.parameters.maxPath, fpSize=self.parameters.fpSize
        )
        fp = rdkit_gen.GetFingerprint(mol)
        return numpy_from_rdkit(fp, dtype=np.int8)


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
        return FittedSklearnScaler(saved_params=saved_params)


@dataclass
class FittedSklearnScaler:
    saved_params: str
    name: Literal["FittedSklearnScaler"] = "FittedSklearnScaler"

    def get_fitted_scaler(self):
        return jsonpickle.loads(self.saved_params)


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
        rdkit_names: Optional[List[str]] = None

    name: Literal["UnscaledPhyschemDescriptors"] = "UnscaledPhyschemDescriptors"
    parameters: Parameters = Parameters()

    def _rdkit_descriptors(self):
        return MolecularDescriptorCalculator(self.parameters.rdkit_names)

    def __post_init__(self):
        # Get RDKit descriptor names.
        if self.parameters.rdkit_names is None:
            self.parameters.rdkit_names = [d[0] for d in Descriptors._descList]

    def calculate_from_mol(self, mol: Chem.Mol):
        d = self._rdkit_descriptors().CalcDescriptors(mol)
        if np.isnan(d).any():
            return None
        else:
            return np.array(d)


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
        jazzy_names: Optional[List[str]] = None
        jazzy_filters: Optional[Dict] = None

    name: Literal["UnscaledJazzyDescriptors"] = "UnscaledJazzyDescriptors"
    parameters: Parameters = Parameters()

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

        if self._exceeds_descriptor_threshold(smi):
            """Raise JazzyError if descriptor input is expected to be slow"""
            raise JazzyError
        else:
            """Return the MMFF94 vector"""
            return molecular_vector_from_smiles(smi, minimisation_method="MMFF94")

    def __post_init__(self):
        # Get Jazzy descriptor names.
        if self.parameters.jazzy_names is None:
            self.parameters.jazzy_names = sorted(
                list(molecular_vector_from_smiles("C", minimisation_method="MMFF94"))
            )
        if self.parameters.jazzy_filters is None:
            self.parameters.jazzy_filters = {
                "NumHAcceptors": 25,
                "NumHDonors": 25,
                "MolWt": 1000,
            }

    def calculate_from_smi(self, smi: str) -> np.ndarray | None:
        try:
            d = self._jazzy_descriptors(smi)
            d = [d[jazzy_name] for jazzy_name in self.parameters.jazzy_names]
        except (JazzyError, TypeError):
            return None
        if np.isnan(d).any():
            return None
        else:
            return np.array(d)


@dataclass
class UnscaledZScalesDescriptors(MolDescriptor):
    """Unscaled Z-Scales.

    Compute the Z-scales of a peptide SMILES.
    """

    @apischema.type_name("UnscaledZScalesDescParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["UnscaledZScalesDescriptors"] = "UnscaledZScalesDescriptors"
    parameters: Parameters = Parameters()

    def calculate_from_smi(self, smi: str) -> list[str | list[None]] | None:
        from chemistry_adapters import AminoAcidAdapter
        from peptides import Peptide

        try:
            sequence = AminoAcidAdapter().convert_smiles_to_amino_acid_sequence(smi)
            fp = list(Peptide(sequence).z_scales())
        except KeyError:
            return None  # SMILES to aa sequence failed
        return fp


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
        file: Annotated[
            str,
            schema(
                title="file",
                description="Name of the CSV containing precomputed descriptors",
            ),
        ] = field(default=None)
        input_column: Annotated[
            str,
            schema(
                title="Input Column",
                description="Name of input column with SMILES strings",
            ),
        ] = field(default=None)
        response_column: Annotated[
            str,
            schema(
                title="Response column",
                description="Name of response column with the comma-separated vectors that the model will use as "
                "pre-computed descriptors",
            ),
        ] = field(default=None)

    name: Literal["PrecomputedDescriptorFromFile"]
    parameters: Parameters

    def calculate_from_smi(self, smi: str) -> np.ndarray:
        file = self.parameters.file
        df = pd.read_csv(file, skipinitialspace=True)
        rows = df[df[self.parameters.input_column].str.replace(">>", ".") == smi][
            [self.parameters.input_column, self.parameters.response_column]
        ].drop_duplicates()
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


@dataclass
class SmilesFromFile(MolDescriptor):
    """Smiles as descriptors (for ChemProp).

    ChemProp optimisation runs require either this or SmilesAndSideInfoFromFile descriptor to be selected.
    This setting allows the SMILES to pass through to the ChemProp package.
    """

    @apischema.type_name("SmilesFromFileParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["SmilesFromFile"]
    parameters: Parameters = Parameters()

    def calculate_from_smi(self, smi: str) -> list[str | list[None]] | None:
        # Handle RDkit errors here to avoid poor handling by ChemProp
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        # None is used as placeholder to indicate no side information is used
        return [smi, [None]]


@dataclass
class SmilesAndSideInfoFromFile(MolDescriptor):
    """SMILES & side information descriptors (for ChemProp).

    ChemProp optimisation requires either these or SmilesFromFile descriptors. This descriptor allows SMILES to pass
    through to ChemProp, _and_ for side information to be supplied as auxiliary tasks.\n

    Side information can take the form of any vector (continuous or binary) which describe input compounds. All tasks
    are learnt in a multi-task manner to improve main-task (task of intent) predictions. Side information can boost
    performance since their contribution to network loss can lead to improved learnt molecular representations.\n

    Optimal side information weighting (how much auxiliary tasks contribute to network loss) is also
    an (optional) learned parameter during optimisation.\n

    Similar to PrecomputedDescriptorFromFile, CSV inputs for this descriptor should contain a SMILES column of
    input molecules. All vectors in the remaining columns are used as user-derived side-information (i.e: be cautious
    to only upload a CSV with side information tasks in columns since _all_ are used)\n

    (see https://ruder.io/multi-task/index.html#auxiliarytasks for details).
    """

    @apischema.type_name("SmilesAndSideInfoFromFileParams")
    @dataclass
    class Parameters:
        @type_name("SmilesAndSideInfoFromFileAux_Weight_Pc")
        @dataclass
        class Aux_Weight_Pc:
            low: int = field(default=100, metadata=schema(min=0, max=100))
            high: int = field(default=100, metadata=schema(min=0, max=100))
            q: int = field(default=20, metadata=schema(min=1))

        file: Annotated[
            str,
            schema(
                title="file",
                description="Name of the CSV containing precomputed side-info descriptors",
            ),
        ] = field(default=None)
        input_column: Annotated[
            str,
            schema(
                title="Input Column",
                description="Name of input column with SMILES strings",
            ),
        ] = field(default=None)
        aux_weight_pc: Annotated[
            Aux_Weight_Pc,
            schema(
                title="Auxiliary weight percentage",
                description="How much (%) auxiliary tasks (side information) contribute (%)"
                "to the loss function optimised during training. The larger the number, "
                "the larger the weight of side information.",
            ),
        ] = Aux_Weight_Pc()

    name: Literal["SmilesAndSideInfoFromFile"]
    parameters: Parameters

    def calculate_from_smi(self, smi: str) -> list[str, Any] | None:
        # Handle RDkit errors here to avoid poor handling by ChemProp
        mol = mol_from_smi(smi)
        if mol is None:
            return None
        file = self.parameters.file
        df = pd.read_csv(file, skipinitialspace=True)
        rows = df[df[self.parameters.input_column] == smi]
        if len(rows) < 1:
            raise ValueError(
                f"Could not find descriptor for {smi} in file {self.parameters.file}."
            )
        if len(rows) > 1:
            logger.warning(
                f"Multiple descriptors found for {smi}, taking the first one."
            )
        descriptor = rows.drop(self.parameters.input_column, axis=1).values[:1]
        return [smi, descriptor]


SmilesBasedDescriptor = Union[SmilesFromFile, SmilesAndSideInfoFromFile]

AnyUnscaledDescriptor = Union[
    Avalon,
    ECFP,
    ECFP_counts,
    PathFP,
    MACCS_keys,
    PrecomputedDescriptorFromFile,
    UnscaledPhyschemDescriptors,
    UnscaledJazzyDescriptors,
    UnscaledZScalesDescriptors,
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
            self._ensure_scaler_is_fitted(cache=cache)
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
            self._ensure_scaler_is_fitted()

    def _ensure_scaler_is_fitted(self, cache=None):
        if isinstance(self.parameters.scaler, UnfittedSklearnScaler):
            self.parameters.scaler = self.parameters.scaler.get_fitted_scaler_for_fp(
                self.parameters.descriptor, cache=cache
            )

        # Cache unpickled sklearn scaler.
        if self._scaler is None:
            self._scaler = self.parameters.scaler.get_fitted_scaler()

    def calculate_from_smi(self, smi: str, cache=None) -> Optional[np.ndarray]:
        self._ensure_scaler_is_fitted()

        if cache is not None:
            cache_desc = cache.cache(self.parameters.descriptor.calculate_from_smi)
            desc = cache_desc(smi)
        else:
            desc = self.parameters.descriptor.calculate_from_smi(smi)  # 1d array.

        if desc is None:
            return None

        # Scikit-learn scaler takes 2d array, one sample per row. Reshape.
        desc_2d = np.array(desc).reshape(1, -1)  # single sample = single row.

        scaled = self._scaler.transform(desc_2d)

        scaled_1d = scaled[0, :]  # return as 1d array.

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
        rdkit_names: Optional[List[str]] = None
        scaler: Union[
            FittedSklearnScaler,
            UnfittedSklearnScaler,
        ] = UnfittedSklearnScaler()
        descriptor: AnyUnscaledDescriptor = UnscaledPhyschemDescriptors

    parameters: Parameters = Parameters()
    name: Literal["PhyschemDescriptors"] = "PhyschemDescriptors"

    def __post_init__(self):
        # Get RDKit descriptor names.
        if self.parameters.rdkit_names is None:
            self.parameters.rdkit_names = [d[0] for d in Descriptors._descList]
        self.parameters.descriptor = UnscaledPhyschemDescriptors.new(
            rdkit_names=self.parameters.rdkit_names
        )
        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("PhyschemDescriptors scaling failed")


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
        jazzy_names: Optional[List[str]] = None
        jazzy_filters: Optional[Dict] = None
        scaler: Union[
            FittedSklearnScaler,
            UnfittedSklearnScaler,
        ] = UnfittedSklearnScaler()
        descriptor: AnyUnscaledDescriptor = UnscaledJazzyDescriptors

    name: Literal["JazzyDescriptors"] = "JazzyDescriptors"
    parameters: Parameters = Parameters()

    def __post_init__(self):
        # Get Jazzy descriptor names.
        if self.parameters.jazzy_names is None:
            self.parameters.jazzy_names = sorted(
                list(molecular_vector_from_smiles("C", minimisation_method="MMFF94"))
            )

        if self.parameters.jazzy_filters is None:
            self.parameters.jazzy_filters = {
                "NumHAcceptors": 25,
                "NumHDonors": 25,
                "MolWt": 1200,
            }

        self.parameters.descriptor = UnscaledJazzyDescriptors.new(
            jazzy_names=self.parameters.jazzy_names,
            jazzy_filters=self.parameters.jazzy_filters,
        )

        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("JazzyDescriptors scaling failed")


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
        scaler: Union[
            FittedSklearnScaler,
            UnfittedSklearnScaler,
        ] = UnfittedSklearnScaler()
        descriptor: AnyUnscaledDescriptor = UnscaledZScalesDescriptors

    name: Literal["ZScalesDescriptors"] = "ZScalesDescriptors"
    parameters: Parameters = Parameters()

    def __post_init__(self):
        self.parameters.descriptor = UnscaledZScalesDescriptors.new()

        if (
            isinstance(self.parameters.scaler, UnfittedSklearnScaler)
            and self.parameters.scaler.mol_data.file_path is not None
        ):
            try:
                self._ensure_scaler_is_fitted()
            except ScalingFittingError:
                logger.warning("ZScales scaling failed")


CompositeCompatibleDescriptor = Union[
    AnyUnscaledDescriptor,
    ScaledDescriptor,
    PhyschemDescriptors,
    JazzyDescriptors,
    ZScalesDescriptors,
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

    parameters: Parameters
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

AnyDescriptor = Union[AnyChemPropIncompatible, SmilesBasedDescriptor]


@dataclass
class CanonicalSmiles(MolDescriptor):
    """Canonical Smiles for use in utility functions (not for user selection)."""

    @apischema.type_name("CanonicalSmilesParams")
    @dataclass
    class Parameters:
        pass

    name: Literal["CanonicalSmiles"]
    parameters: Parameters = Parameters()

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

    name: Literal["Scaffold"]
    parameters: Parameters = Parameters()

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

    name: Literal["GenericScaffold"]
    parameters: Parameters = Parameters()

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

    name: Literal["ValidDescriptor"]
    parameters: Parameters = Parameters()

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
            if d is None:
                failed_idx.append(d_idx)
            elif isinstance(descriptor, SmilesBasedDescriptor):
                list_of_arrays.append(d)
            elif not pd.isnull(np.array(d)).any():
                list_of_arrays.append(d)
            else:
                failed_idx.append(d_idx)
        if len(list_of_arrays) > 0:
            list_of_arrays = np.stack(list_of_arrays, axis=0)
        return list_of_arrays, failed_idx
    else:
        return descriptors
