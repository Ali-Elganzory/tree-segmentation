import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor

from src.model import Models
from src.trainer import LossFn
from src.dataset import Datasets


def command_factory(
    name: str,
    dataset: Datasets,
    epochs: int,
    loss: LossFn,
    weighted_loss: bool = True,
    batch_norm: bool = True,
    skip_batch_norm_on_trans_conv: bool = True,
    model: Models = Models.dino,
    batch_size: int = 16,
):
    return (
        f"python run.py --dataset {dataset.value} --model {model.value} --batch-size {batch_size} --epochs {epochs} --loss {loss.value} {'--no-weighted-loss' if not weighted_loss else ''}"
        f" {' --no-batch-norm' if not batch_norm else ''} {'--no-skip-batch-norm-on-trans-conv' if not skip_batch_norm_on_trans_conv else ''}"
        f" --results-file results/{name}.csv --save-to checkpoints/{name}.pt 2>&1 | tee logs/{name}.log"
    )


NUM_EPOCHS = 50

voc_commands = [
    command_factory(
        "voc-cross_entropy",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=False,
        batch_norm=False,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "voc-cross_entropy-weighted_loss",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=False,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "voc-cross_entropy-weighted_loss-batch_norm",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=True,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "voc-cross_entropy-weighted_loss-batch_norm-skip_tconv_batch_norm",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=True,
        skip_batch_norm_on_trans_conv=True,
    ),
    command_factory(
        "voc-focal",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.focal,
    ),
    command_factory(
        "voc-focal_tversky",
        Datasets.voc,
        NUM_EPOCHS,
        LossFn.focal_tversky,
    ),
]

trees_commands = [
    command_factory(
        "trees-cross_entropy",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=False,
        batch_norm=False,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "trees-cross_entropy-weighted_loss",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=False,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "trees-cross_entropy-weighted_loss-batch_norm",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=True,
        skip_batch_norm_on_trans_conv=False,
    ),
    command_factory(
        "trees-cross_entropy-weighted_loss-batch_norm-skip_tconv_batch_norm",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        weighted_loss=True,
        batch_norm=True,
        skip_batch_norm_on_trans_conv=True,
    ),
    command_factory(
        "trees-focal",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.focal,
    ),
    command_factory(
        "trees-focal_tversky",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.focal_tversky,
    ),
    command_factory(
        "trees-b32",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        batch_size=32,
    ),
    command_factory(
        "trees-b64",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        batch_size=64,
    ),
    command_factory(
        "trees-deeplab-b16",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        batch_size=16,
    ),
    command_factory(
        "trees-deeplab-b32",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        batch_size=32,
    ),
    command_factory(
        "trees-deeplab-b64",
        Datasets.trees,
        NUM_EPOCHS,
        LossFn.cross_entropy,
        batch_size=64,
    ),
]

commands = [
    *voc_commands,
    *trees_commands,
]

gpu_available = [True, True, True, True]


def run_command(command: str, gpu_id: int):
    command = f"zsh -c '. ~/.zshrc && mamba activate dl && CUDA_VISIBLE_DEVICES={gpu_id} {command}'"
    os.system(command)
    gpu_available[gpu_id] = True


def run_experiments(commands):
    with ThreadPoolExecutor(max_workers=4) as executor:
        for command in commands:
            while not any(gpu_available):
                sleep(1)

            gpu_id = gpu_available.index(True)
            gpu_available[gpu_id] = False
            executor.submit(run_command, command, gpu_id)


if __name__ == "__main__":
    run_experiments(commands)
