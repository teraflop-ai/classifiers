import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_labels=1, binary=False):
        super().__init__()
        self.binary = binary
        self.probe = nn.Linear(in_dim, 1 if binary else num_labels)

    def forward(self, x):
        out = self.probe(x)
        return torch.sigmoid(out).squeeze(-1) if self.binary else out
