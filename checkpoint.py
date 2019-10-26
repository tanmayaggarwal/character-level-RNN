# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def checkpoint(net):
    # change the name, for saving multiple files
    model_name = 'rnn_x_epoch.net'

    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict':net.state_dict(),
                  'tokens': net.chars}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)

    return f
