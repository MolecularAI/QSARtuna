# File: tests/conftest.py
import pytest
import shutil

@pytest.fixture
def clean_shared_datadir(shared_datadir):
    yield shared_datadir
    # Cleanup after the test
    shutil.rmtree(shared_datadir, ignore_errors=True)