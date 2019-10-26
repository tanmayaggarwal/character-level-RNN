# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# three steps:
# 1. discard some of the text so we only have completely full mini-batches
# 2. split arr into N batches
# 3. iterate through the arrays to get mini-batches


def get_batches(arr, batch_size, seq_length):
    # function is a generator that returns batches of size
    # batch_size x seq_length from arr.

    # Arguments
    # ---------
    # arr: Array that we want to make batches from
    # batch_size: the number of sequences per batch
    # seq_length: number of encoded chars in a sequence

    # number of batches we can make
    n_batches = int(len(arr) / (batch_size * seq_length))

    # discard characters to make full batches
    slice_index = int(n_batches * batch_size * seq_length)
    arr = arr[:slice_index]

    # reshape into batch_size rows
    size = tuple((batch_size,-1))
    arr = arr.reshape(size)

    # iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # The features
        mini_batch_index = int(batch_size*seq_length)
        x = arr[n:mini_batch_index]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

    return x, y
