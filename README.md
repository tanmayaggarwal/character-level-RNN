# character-level-RNN

Character level RNN using LSTMs with PyTorch

This application builds a character-level RNN using LSTMs with PyTorch <br/>
The network will train character by character on some text, then generate new text character by character. <br/>
Input: text from any text file <br/>
Output: model will generate new text based on the text from input file <br/>

There are four main steps included in this application:<br/>
a. Load in text data<br/>
b. Pre-process the data, encoding characters as integers and creating one-hot input vectors<br/>
c. Define an RNN that predicts the next character when given an input sequence<br/>
d. Train the RNN and use it to generate new text<br/>

Standard imports:

import numpy as np <br/>
import torch <br/>
from torch import nn <br/>
import torch.nn.functional as F <br/>
