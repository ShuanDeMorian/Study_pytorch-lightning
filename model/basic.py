from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from .cnn import CNN
from .resnet import ResNet


class LiNet(pl.LightningModule):
    def __init__(
        self,
        seed=1234,
        lr=1e-2,
        hidden_size=64,
        dropout_rate=0.1,
        model_type="cnn",
        optim="adam",
    ):
        super().__init__()
        # This fixes a seed for all the modules used by pytorch-lightning
        # Note: using random.seed(seed) is not enough, there are multiple
        # other seeds like hash seed, seed for numpy etc.
        self.seed = seed
        pl.seed_everything(self.seed)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.functional.softmax

        self.model_type = model_type
        self.optim = optim
        if self.model_type == "resnet":
            self.nn = ResNet(hidden_size, dropout_rate)
        elif self.model_type == "cnn":
            self.nn = CNN()

    def forward(self, x):
        logits = self.nn(x)
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
        x, y = batch
        # x : b x 1 x 28 x 28
        # y : label

        # 1 forward
        logits = self(x)

        # 2 compute the objective function
        J = self.loss(logits, y)
        acc = accuracy(self.softmax(logits, dim=1), y)
        self.log("train_loss", J, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return {"loss": J, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x : b x 1 x 28 x 28
        # y : label

        # 1 forward
        logits = self(x)

        # 2 compute the objective function
        J = self.loss(logits, y)
        acc = accuracy(self.softmax(logits, dim=1), y)
        self.log("val_loss", J, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"loss": J, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # x : b x 1 x 28 x 28
        # y : label

        # 1 forward
        logits = self(x)

        # 2 compute the objective function
        J = self.loss(logits, y)
        acc = accuracy(self.softmax(logits, dim=1), y)
        self.log("test_loss", J, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return {"loss": J, "acc": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        # x : b x 1 x 28 x 28
        # y : label

        # 1 forward
        logits = self(x).detach().cpu()
        y = y.detach().cpu()
        return {
            "logit": logits.tolist(),
            "predict": logits.argmax(dim=1).tolist(),
            "label": y.tolist(),
        }
