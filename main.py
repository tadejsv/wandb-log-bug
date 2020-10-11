import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from data import MNISTDataModule
from model import MNISTClassifier

@hydra.main(config_name="config")
def run_train(cfg: DictConfig) -> None:

    data_module = MNISTDataModule()
    model = MNISTClassifier(lr=cfg.learning_rate)

    wandb = WandbLogger(offline=True)
    trainer = pl.Trainer(max_epochs=1, logger=wandb)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    run_train()
