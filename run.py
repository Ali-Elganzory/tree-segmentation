from pathlib import Path

import torch
from typer import Typer

from src import logging
from src.trainer import Trainer, LossFn
from src.dataset import TreesDataLoaders
from src.model import DinoV2Model as Model

app = Typer()


def setup(verbose: bool = True):
    # Logging
    logging.enable(logging.LogLevel.INFO if verbose else logging.LogLevel.WARNING)

    # Distributed
    # distributed.enable()

    print(f"Started", force=True)


@app.command()
def train(
    folder: Path = Path("data/quebec_trees_dataset_2021-09-02"),
    epochs: int = 1,
    verbose: bool = True,
):
    setup(verbose)

    # Dataset
    dataloaders = TreesDataLoaders(
        folder=folder,
        batch_size=32,
        num_workers=8,
        transform=Model.transforms,
        augmentations=Model.augmentations,
    )

    # Model
    model = Model(num_classes=len(dataloaders.labels))

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=LossFn.focal_tversky,
        results_file="results.csv",
    )
    trainer.train(
        epochs=epochs,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
    )

    # Save
    trainer.save()


if __name__ == "__main__":
    app()
