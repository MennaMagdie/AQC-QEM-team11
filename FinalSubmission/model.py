import torch
import torch.nn as nn

class QuantumErrorMitigator(nn.Module):
    def __init__(self):
        super(QuantumErrorMitigator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(20, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 16),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)