from pathlib import Path

import torch
from typer import Typer

from src import logging
from src.dataset import Datasets
from src.model import DinoV2Model as Model
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
    loss: LossFn = LossFn.cross_entropy.value,
    weighted_loss: bool = True,
    batch_norm: bool = True,
    skip_batch_norm_on_trans_conv: bool = False,
    epochs: int = 1,
    results_file: Path = Path("results.csv"),
    verbose: bool = True,
):
    setup(verbose)

    # Dataset
    dataloaders = dataset.factory(
        batch_size=16,
        num_workers=8,
        transform=Model.transforms,
        augmentations=Model.augmentations,
    )

    # Model
    model = Model(
        num_classes=21,
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
        labels=torch.tensor(range(21)) if weighted_loss else None,
    )

    # Save
    trainer.save()


if __name__ == "__main__":
    app()
