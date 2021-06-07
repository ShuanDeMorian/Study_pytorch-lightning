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


class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 1e-2
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.functional.softmax

    #############################################################################
    # For multi gpu training
    def prepare_data(self):
        datasets.MNIST(
            "data", train=True, download=True, transform=transforms.ToTensor()
        )

    def setup(self, stage):
        dataset = datasets.MNIST(
            "data", train=True, download=False, transform=transforms.ToTensor()
        )
        self.train_split, self.val_split = random_split(dataset, [55000, 5000])

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
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits

    def configure_optimizers(self):
        # return optim.SGD(self.parameters(), lr=self.lr)
        # return optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, "max"),
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


model = ResNet()
lr_monitor = LearningRateMonitor(logging_interval="step")
# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
trainer = pl.Trainer(
    max_epochs=1000,
    gpus=1,
    stochastic_weight_avg=True,
    auto_lr_find=True,
    callbacks=[lr_monitor],
)
trainer.tune(model)
trainer.fit(model)
