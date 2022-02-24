import glob
import json

import pytest
from apischema import deserialize

from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import OptimizationConfig
from optunaz.utils.files_paths import attach_root_path


@pytest.mark.parametrize(
    "filename", glob.glob(f"{attach_root_path('examples/optimization')}/*.json")
)
def test_optimization_json(filename):
    with open(filename, "rt") as f:
        data = json.load(f)
    deserialize(OptimizationConfig, data)


@pytest.mark.parametrize(
    "filename",
    glob.glob(f"{attach_root_path('examples/optimization_templates')}/*.json"),
)
def test_optimization_templ_json(filename):
    with open(filename, "rt") as f:
        data = json.load(f)
    deserialize(OptimizationConfig, data)


@pytest.mark.parametrize(
    "filename", glob.glob(f"{attach_root_path('examples/building')}/*.json")
)
def test_build_json(filename):
    with open(filename, "rt") as f:
        data = json.load(f)
    deserialize(BuildConfig, data)
