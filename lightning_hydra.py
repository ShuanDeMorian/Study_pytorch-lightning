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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import hydra
from omegaconf import DictConfig


class LiNet(pl.LightningModule):
    def __init__(
        self,
        data_dir="data",
        seed=1234,
        batch_size=32,
        num_workers=4,
        lr=1e-2,
        hidden_size=64,
        dropout_rate=0.1,
        model_type="cnn",
        optim="adam",
    ):
        super().__init__()
        self.data_dir = data_dir
        # This fixes a seed for all the modules used by pytorch-lightning
        # Note: using random.seed(seed) is not enough, there are multiple
        # other seeds like hash seed, seed for numpy etc.
        self.seed = seed
        pl.seed_everything(self.seed)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.functional.softmax

        self.model_type = model_type
        self.optim = optim
        if self.model_type == "resnet":
            self.l1 = nn.Linear(28 * 28, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, hidden_size)
            self.final = nn.Linear(hidden_size, 10)
            self.do = nn.Dropout(dropout_rate)
        elif self.model_type == "cnn":
            self.layer = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

    #############################################################################
    # For multi gpu training
    def prepare_data(self):
        datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            target_transform=None,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ]
            ),
        )
        datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            target_transform=None,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ]
            ),
        )

    def setup(self, stage):
        self.train_split = datasets.MNIST(
            self.data_dir,
            train=True,
            download=False,
            target_transform=None,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ]
            ),
        )
        self.val_split = datasets.MNIST(
            self.data_dir,
            train=False,
            download=False,
            target_transform=None,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ]
            ),
        )
        # self.train_split, self.val_split = random_split(
        #     dataset, [55000, 5000], generator=torch.Generator().manual_seed(self.seed)
        # )

    #############################################################################
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_split, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_split, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return val_loader

    def forward(self, x):
        if self.model_type == "resnet":
            h1 = nn.functional.relu(self.l1(x))
            h2 = nn.functional.relu(self.l2(h1))
            do = self.do(h1 + h2)
            h3 = nn.functional.relu(self.l3(do))
            do = self.do(h2 + h3)
            logits = self.final(do)
        elif self.model_type == "cnn":
            batch_size = x.shape[0]
            x = x.view(batch_size, 1, 28, 28)
            out = self.layer(x)
            out = out.view(batch_size, -1)
            logits = self.fc_layer(out)
        return logits

    def configure_optimizers(self):
        if self.optim == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optim == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, threshold=0.1, patience=1, mode="max"
                ),
                "monitor": "val_acc",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch  # y : label

        # x : b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        logits = self(x)

        # 2 compute the objective function
        J = self.loss(logits, y)
        acc = accuracy(self.softmax(logits), y)
        self.log("train_loss", J, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return {"loss": J, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch  # y : label

        # x : b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        logits = self(x)

        # 2 compute the objective function
        J = self.loss(logits, y)
        acc = accuracy(self.softmax(logits), y)
        self.log("val_loss", J, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"loss": J, "acc": acc}


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    gpus = 1
    num_workers = gpus * 4

    # Init model
    model = LiNet(
        data_dir=config.data_dir,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=num_workers,
        lr=config.lr,
        hidden_size=config.hidden_size,
        dropout_rate=config.dropout_rate,
        model_type=config.model_type,
        optim=config.optim,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        gpus=gpus,
        stochastic_weight_avg=config.swa,
        auto_lr_find=config.auto_lr,
        callbacks=[lr_monitor],
    )
    if config.auto_lr:
        trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
