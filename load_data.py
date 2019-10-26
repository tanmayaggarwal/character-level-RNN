# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# open text file and read in data as text

def load_data(input_source):
    with open(input_source, 'r') as f:
        text = f.read()

    # check out the first 100 characters
    print(text[:100])

    return text
