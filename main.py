# This application builds a character-level RNN using LSTMs with PyTorch
# The network will train character by character on some text, then generate new text
# character by character.
# Input: text from any book
# Output: model will generate new text based on the text from input book

# There are four main steps included in this application:
# a. Load in text data
# b. Pre-process the data, encoding characters as integers and creating one-hot input vectors
# c. Define an RNN that predicts the next character when given an input sequence
# d. Train the RNN and use it to generate new text

# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# load in data

input_source = 'data/input_source.txt' # change source as needed
from load_data import load_data
text = load_data(input_source)

# Tokenization (converting text to and from integers)

from tokenization import tokenization
chars, encoded = tokenization(text)

# defining the one hot encoding function; parameters: (array, n_labels)
from one_hot_encode import one_hot_encode

# making training mini-batches
from get_batches import get_batches

# test implementation
batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# printing out the first 10 items in a sequence
print ('x\n', x[:10, :10])
print ('\ny\n', y[:10, :10])

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

# define the network with PyTorch
from CharRNN import CharRNN

# train the function
from train import train

# instantiating the model

# set model hyperparameters
n_hidden = 256
n_layers = 2

net = CharRNN(chars, n_hidden, n_layers)
print(net)

# set training hyperparameters
batch_size = 10
seq_length = 100
n_epochs = 20

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

# save the model
from checkpoint import checkpoint
f = checkpoint(net)

# predict the next character
from predict import predict

# priming and generating text
from sample import sample
print(sample(net, 1000, prime="Hello", top_k=5))
