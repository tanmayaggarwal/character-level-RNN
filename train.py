# standard imports

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def train(net, data, epochs=1, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    # Training the network

    # Arguments
    # ---------

    # net: CharRNN network
    # data: text data to train the network
    # epochs: number of epochs to train
    # batch_size: number of mini-sequences per mini-batch
    # seq_length: number of character steps per mini-batch
    # lr: learning rate
    # clip: gradient clipping
    # val_frac: fraction of data to hold out for validation
    # print_every: number of steps for printing training and validation loss

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if (train_on_gpu):
        net.cuda()

    counter=0
    n_chars = len(net.chars)

    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # one-hot encode data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # create new variables for the hidden state, to avoid backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            #clip_grad_norm prevents exploding gradient problem
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss statistics
            if counter % print_every == 0:
                # get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # one-hot encode the data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # create new variables for the hidden state to avoid backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)

                    val_loss = criterion(output, targets.view(batch_size*seq_length))

                    val_losses.append(val_loss.item())

                net.train()

                print("Epoch: {}/{} ...".format(e+1, epochs),
                      "Step: {} ...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}...".format(np.mean(val_losses)))

    return
