from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

from optunaz import explainability
from optunaz.model_writer import Predictor
from optunaz.descriptors import ECFP, ECFP_counts


@dataclass
class Model(Predictor):
    def predict(self, xs):
        return np.sum(xs)


@dataclass
class Estimator:
    pass


def test_ecfp():
    mol = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()

    fp = mfpgen.GetFingerprint(mol, additionalOutput=ao)
    descriptor = ECFP.new(radius=2)
    info = explainability.get_ecfp_fpinfo(mol, descriptor)
    assert len(info) == len(ao.GetBitInfoMap())
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
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()

    fp = mfpgen.GetCountFingerprintAsNumPy(mol, additionalOutput=ao)
    descriptor = ECFP_counts.new(radius=2)
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
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()
    fp = mfpgen.GetFingerprint(mol, additionalOutput=ao)

    descriptor = ECFP.new(radius=2)
    m0 = Model()
    m0.X_ = np.array([fp.ToList()])
    m0.train_smiles_ = [train_smiles]
    explained_feats = explainability.explain_ECFP_PathFP(2048, m0, descriptor)
    assert len(explained_feats) == 2048
    assert explained_feats[389] == "OC"


def test_explain_ecfp_count():
    train_smiles = "O=C(C)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(train_smiles)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()
    fp = mfpgen.GetCountFingerprint(mol, additionalOutput=ao)
    descriptor = ECFP_counts.new(radius=2)
    m0 = Model()
    m0.X_ = np.array([fp.ToList()])
    m0.train_smiles_ = [train_smiles]
    explained_feats = explainability.explain_ECFP_PathFP(2048, m0, descriptor)
    assert len(explained_feats) == 2048
    assert explained_feats[1199] == "c(cc)cc"
