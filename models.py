import torch.nn as nn
from nltk.tokenize import word_tokenize
import numpy as np


class Encoder(nn.Module):
    # Gebruik de glove vectors en de vocab met een nn.Embededding Layer die in elk encoder model moet zitten
    def __init__(self):
        pass
    
    def forward(self):
        pass

class Classifier(nn.Module):
    pass
    # een aparte class (de Classifier) die 1 van deze encoder pakt op basis van de arguments/parameters in de command line en de classificatie doet zoals in het plaatje in de assignment


# class LSTM(nn.Module):
#     def __init__(self, parameters):
#         pass

# class BiLSTM(nn.Module):
#     def __init__(self, parameters):
#         pass

# class BiLSTM_pooling(nn.Module):
#     def __init__(self, parameters):
#         pass