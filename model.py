import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl


class MNISTClassifier(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

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
        
    def training_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        loss = F.nll_loss(output, target)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        loss = F.nll_loss(output, target)

        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.lr)

        return optimizer