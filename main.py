import os
import logging
log = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Module, Linear, functional as F

import wandb

##########################################
# Data

class FakeData(Dataset):
    def __getitem__(self, idx):
        return torch.rand(200, dtype=torch.float), torch.randint(2, (1,))[0]

    def __len__(self):
        return 5000

def train_dataloader():
    return DataLoader(FakeData(), batch_size=16, num_workers=os.cpu_count())

#########################################
# Model

class MNISTClassifier(Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(200, 10)

    def forward(self, x: torch.Tensor):
        return F.log_softmax(self.lin1(x), dim=1)

#########################################
# Train

@hydra.main(config_name="config")
def run_train(cfg: DictConfig) -> None:

    os.environ["WANDB_MODE"] = "dryrun"

    train_loader = train_dataloader()
    model = MNISTClassifier()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    # Changing reinit to False removes the error
    wandb_log = wandb.init(config=cfg, reinit=True)
    wandb_log.watch(model)
    
    log.info('Starting training')

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            wandb_log.log({"loss": loss})
            log.info('loss %s', loss)
            
    log.info('Finished training')


if __name__ == "__main__":
    run_train()
