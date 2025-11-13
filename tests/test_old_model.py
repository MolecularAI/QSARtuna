import pytest
import dill

@pytest.fixture
def old_model(clean_shared_datadir):
    return str(clean_shared_datadir / "rdkit_dependant_model.pkl")

def test_load(old_model):
    with open(old_model, "rb") as fileobj:
        model = dill.load(fileobj)
        preds = model.predict_from_smiles("CCO")