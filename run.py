from pathlib import Path

import torch
from typer import Typer

from src import logging
from src.model import Models
from src.dataset import Datasets
from src.trainer import Trainer, LossFn, Optimizer, LR_Scheduler

app = Typer()


def setup(verbose: bool = True):
    # Logging
    logging.enable(logging.LogLevel.INFO if verbose else logging.LogLevel.WARNING)

    # Distributed
    # distributed.enable()

    print(f"Started", force=True)


@app.command()
def train(
    dataset: Datasets = Datasets.voc.value,
    model: Models = Models.dino.value,
    loss: LossFn = LossFn.cross_entropy.value,
    weighted_loss: bool = True,
    batch_norm: bool = True,
    skip_batch_norm_on_trans_conv: bool = True,
    epochs: int = 1,
    batch_size: int = 16,
    background_threshold: float = 0.75,
    results_file: Path = Path("results/results.csv"),
    save_to: Path = Path("checkpoints/model.pt"),
    verbose: bool = True,
):
    setup(verbose)

    # Dataset
    kwargs = (
        {
            "background_threshold": background_threshold,
        }
        if dataset == Datasets.trees.value
        else {}
    )
    dataloaders = dataset.factory(
        batch_size=batch_size,
        num_workers=8,
        transform=model.factory.transforms,
        augmentations=model.factory.augmentations,
        **kwargs,
    )

    # Model
    model = model.factory(
        num_classes=dataloaders.num_classes,
        batch_norm=batch_norm,
        skip_batch_norm_on_trans_conv=skip_batch_norm_on_trans_conv,
    )

    # Train
    trainer = Trainer(
        model=model,
        results_file=results_file,
        # Hyperparameters
        optimizer=Optimizer.adamw,
        loss_fn=loss,
        lr_scheduler=LR_Scheduler.step,
        lr=1e-4,
        scheduler_gamma=0.97,
        scheduler_step_size=200,
        scheduler_step_every_epoch=False,
        weight_decay=1e-2,
    )
    trainer.train(
        epochs=epochs,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        labels=torch.tensor(range(dataloaders.num_classes)) if weighted_loss else None,
    )

    # Save
    trainer.save(save_to)


if __name__ == "__main__":
    app()
