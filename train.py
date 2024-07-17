from pathlib import Path

import torch
from typer import Typer

from src import logging
from src import distributed
from src.trainer import Trainer
from src.dataset import TreesDataLoaders
from src.model import DinoV2Model as Model

app = Typer()


@app.command()
def train(
    folder: Path = Path("data/quebec_trees_dataset_2021-09-02"),
    verbose: bool = True,
):
    # Logging
    logging.enable(logging.LogLevel.INFO if verbose else logging.LogLevel.WARNING)

    # Distributed
    # distributed.enable()

    # Dataset
    dataloaders = TreesDataLoaders(
        folder=folder,
        batch_size=32,
        num_workers=4,
        transform=Model.transforms,
        augmentations=Model.augmentations,
    )

    # Model
    model = Model(num_classes=len(dataloaders.labels))

    # Train
    trainer = Trainer(
        model=model,
        device=torch.device("cuda"),
        output_device=torch.device("cuda"),
        results_file="results.csv",
    )
    trainer.train(
        epochs=1,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
    )


if __name__ == "__main__":
    app()
