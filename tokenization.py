# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def tokenization(text):
    # encode the text and map each character to an integer and vice versa

    # create two dictonaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers

    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    # view first 100 characters as integers
    encoded[:100]

    return chars, encoded
