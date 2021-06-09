"""
Pytorch Lightning
    1. model
    2. optimizer
    3. data
    4. training loops
    5. validation loops
"""
import os
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
import hydra
from omegaconf import DictConfig
from datamodule.mnist import MNIST_datamodule
from model.basic import LiNet


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    gpus = 1
    num_workers = gpus * 4

    # Init model
    model = LiNet(
        seed=config.seed,
        lr=config.lr,
        hidden_size=config.hidden_size,
        dropout_rate=config.dropout_rate,
        model_type=config.model_type,
        optim=config.optim,
    )

    # Datamodule
    mnist_data = MNIST_datamodule(
        seed=config.seed,
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
    )

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch:d}-{val_acc:.5f}",
        verbose=False,
        monitor="val_acc",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_acc", patience=5, verbose=True, mode="max"
    )

    # auto_lr_finds : finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        gpus=gpus,
        stochastic_weight_avg=config.swa,
        auto_lr_find=config.auto_lr,
        callbacks=[lr_monitor, checkpoint_callback, early_stopping],
    )
    if config.auto_lr:
        trainer.tune(model, mnist_data)
    trainer.fit(model, mnist_data)

    trainer.test()


if __name__ == "__main__":
    main()
