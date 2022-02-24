# https://stackoverflow.com/a/57290746

import sys

import optunaz
from optunaz import *

import optunaz.config.optconfig
import optunaz.config.buildconfig
import optunaz.descriptors

sys.modules["qptuna"] = optunaz

# Every module that is used in `isinstance(obj, <module.class>)`
# needs to be reassigned.
sys.modules["qptuna.config.optconfig"] = sys.modules["optunaz.config.optconfig"]
sys.modules["qptuna.config.buildconfig"] = sys.modules["optunaz.config.buildconfig"]
sys.modules["qptuna.descriptors"] = sys.modules["optunaz.descriptors"]
