# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def sample(net, size, prime="The", top_k=None):
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    # first off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
