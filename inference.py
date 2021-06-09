import pytorch_lightning as pl
from datamodule.mnist import MNIST_datamodule
import torch
import json

# import hydra
# from omegaconf import DictConfig
from model.basic import LiNet


def main():
    gpus = 1
    num_workers = gpus * 4

    best_pth = "outputs/2021-06-09/13-07-41/lightning_logs/version_0/checkpoints/epoch=6-val_acc=0.99440.ckpt"
    trainer = pl.Trainer(gpus=gpus)
    model = LiNet.load_from_checkpoint(best_pth)

    mnist_data = MNIST_datamodule(
        seed=1234,
        data_dir="data",
        batch_size=100,
        num_workers=num_workers,
        pin_memory=True,
    )

    # trainer.test(model=model, datamodule=mnist_data, verbose=True)
    results = trainer.predict(model=model, datamodule=mnist_data)
    # save results
    res = []
    for result in results:
        for logit, predict, label in zip(
            result["logit"], result["predict"], result["label"]
        ):
            res.append({"logit": logit, "predict": predict, "label": label})

    with open("result/result.json", "w", encoding="utf-8") as json_file:
        json.dump(res, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
