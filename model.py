import torch
from torch import nn
from torch.nn import functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 8, 3, 1)
        self.dropout = nn.Dropout2d(0.25)
        self.fc = nn.Linear(288, 10)
        self.pool = nn.MaxPool2d(3, stride=4)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)

        return output