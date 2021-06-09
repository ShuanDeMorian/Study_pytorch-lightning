import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch


class MNIST_datamodule(pl.LightningDataModule):
    def __init__(
        self, data_dir="data", seed=1234, batch_size=32, num_workers=4, pin_memory=True
    ):
        super().__init__()
        self.seed = seed
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = datasets.MNIST(
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
            self.train_split, self.val_split = random_split(
                dataset,
                [55000, 5000],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in ["test", "predict"] or stage is None:
            self.test_split = datasets.MNIST(
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

    #############################################################################
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader

    def predict_dataloader(self):
        test_loader = DataLoader(
            self.test_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader
