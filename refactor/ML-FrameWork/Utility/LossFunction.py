import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

def TorchMSELoss():
    return torch.nn.MSELoss()