from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.probe = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.probe(x).squeeze(-1)


class MultiClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.probe = nn.Linear(in_dim, num_labels)

    def forward(self, x):
        return self.probe(x)
