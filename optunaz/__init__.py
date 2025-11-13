import os
# Import torch before numpy to avoid segmentation fault on OSX
import torch
import numpy as np

__version__ = "4.0.1"

os.environ["TQDM_DISABLE"] = "1"
