from rdkit import Chem
from rdkit.Chem import AllChem
from optunaz import explainability
from dataclasses import dataclass
import numpy as np
from optunaz.model_writer import Predictor


@dataclass
class Model(Predictor):
    def predict(self, xs):
        return np.sum(xs)


@dataclass
class Descriptor:
    pass


@dataclass
class Parameters:
    pass


@dataclass
class Estimator:
    pass


def test_ecfp():
    mol = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    descriptor = Descriptor()
    descriptor.parameters = Parameters()
    descriptor.parameters.radius = 2
    descriptor.parameters.nBits = 2048
    descriptor.parameters.useFeatures = False
    info = explainability.get_ecfp_fpinfo(mol, descriptor)
    assert len(info) == fp.GetNumOnBits()
    assert all([len(info.get(i)) != None for i in fp.GetOnBits()])

    first_bit = [i for i in info.items() if i[1][0] != 0]
    first_bit_atom = first_bit[0][1][0][0]
    first_bit_radius = first_bit[0][1][0][1]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, first_bit_radius, first_bit_atom)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    feat_smi = Chem.MolToSmiles(
        submol, rootedAtAtom=amap[first_bit_atom], canonical=False
    )
    assert feat_smi == "OC"


def test_ecfp_count():
    mol = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
    descriptor = Descriptor()
    descriptor.parameters = Parameters()
    descriptor.parameters.radius = 2
    descriptor.parameters.nBits = 2048
    descriptor.parameters.useFeatures = True
    info = explainability.get_ecfp_fpinfo(mol, descriptor)

    first_bit = [i for i in info.items() if i[1][0] != 0]
    first_bit_atom = first_bit[0][1][0][0]
    first_bit_radius = first_bit[0][1][0][1]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, first_bit_radius, first_bit_atom)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    feat_smi = Chem.MolToSmiles(
        submol, rootedAtAtom=amap[first_bit_atom], canonical=False
    )
    assert feat_smi == "OC"


def test_explain_ecfp():
    train_smiles = "O=C(C)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(train_smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    descriptor = Descriptor()
    descriptor.parameters = Parameters()
    descriptor.name = "ECFP"
    descriptor.parameters.radius = 2
    descriptor.parameters.nBits = 2048
    descriptor.parameters.useFeatures = False
    m0 = Model()
    m0.X_ = np.array([fp.ToList()])
    m0.train_smiles_ = [train_smiles]
    explained_feats = explainability.explain_ECFP(2048, m0, descriptor)
    assert len(explained_feats) == 2048
    assert explained_feats[389] == "OC"


def test_explain_ecfp_count():
    train_smiles = "O=C(C)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(train_smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
    descriptor = Descriptor()
    descriptor.parameters = Parameters()
    descriptor.name = "ECFP_counts"
    descriptor.parameters.radius = 2
    descriptor.parameters.nBits = 2048
    descriptor.parameters.useFeatures = True
    m0 = Model()
    m0.X_ = np.array([fp.ToList()])
    m0.train_smiles_ = [train_smiles]
    explained_feats = explainability.explain_ECFP(2048, m0, descriptor)
    assert len(explained_feats) == 2048
    assert explained_feats[598] == "c(cc)cc"
