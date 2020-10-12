import os
import logging
log = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig

from torch.optim import Adam
from torch.nn import functional as F

import wandb

from data import train_dataloader
from model import MNISTClassifier


@hydra.main(config_name="config")
def run_train(cfg: DictConfig) -> None:

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
