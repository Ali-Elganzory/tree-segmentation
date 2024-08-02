import os
from enum import Enum
from pathlib import Path
from typing import Callable

import torch
import numpy as np
import pandas as pd
from PIL import Image
from albumentations import Compose
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
from torch.utils.data.dataset import random_split


class Datasets(Enum):
    trees = "trees"
    voc = "voc"

    @property
    def factory(self):
        return {
            Datasets.trees: TreesDataLoaders,
            Datasets.voc: VOCDataLoaders,
        }[self]


class TreesDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        transform: Callable = None,
        background_threshold: float = 0.75,
    ):
        self.folder = folder
        self.transform = transform
        self.images_folder = folder / "images"
        self.masks_folder = folder / "masks"
        self.background_threshold = background_threshold
        self.cache_path = folder / f"cache_{self.background_threshold}.csv"

        # Create cache if it doesn't exist or if the images or masks have been modified
        if self.cache_path.exists() and (
            max(
                os.path.getmtime(self.images_folder),
                os.path.getmtime(self.masks_folder),
            )
            < os.path.getmtime(self.cache_path)
        ):
            # Load cache
            self.cache = pd.read_csv(self.cache_path)
        else:
            # Create cache
            print(f"Creating cache: {self.cache_path}")
            self.cache = pd.DataFrame(columns=["image", "mask"])
            self.cache["image"] = [Path(p.path) for p in os.scandir(self.images_folder)]
            self.cache["mask"] = self.cache["image"].apply(
                lambda x: self.masks_folder / x.name.replace("jpg", "png"),
            )
            # Filter out masks with too much background and their corresponding images
            self.cache = self.cache[
                self.cache["mask"].apply(
                    lambda x: np.mean(
                        np.array(
                            Image.open(x) if x.exists() else Image.new("L", (1, 1))
                        )
                        == 0
                    )
                    < self.background_threshold
                )
            ]
            self.cache.to_csv(self.cache_path, index=False)
            print(f"Cache created: {self.cache_path}")

        # Load data from cache
        self.data = self.cache["image"].apply(Path).values
        self.target = self.cache["mask"].apply(Path).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filepath = self.data[idx]
        mask_filepath = self.target[idx]

        if not image_filepath.exists():
            raise FileNotFoundError(image_filepath)

        image = Image.open(image_filepath)

        if not mask_filepath.exists():
            mask = Image.new("L", image.size)
        else:
            mask = Image.open(mask_filepath)

        return image, mask


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        image, mask = np.array(image), np.array(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            if isinstance(transformed, dict):
                image, mask = transformed["image"], transformed["mask"]
            else:
                image, mask = transformed

        return image, mask.type(torch.LongTensor)


class TreesDataLoaders:
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        transform: Callable = None,
        augmentations: Callable = None,
        folder: Path = Path("data/quebec_trees_dataset_2021-09-02"),
        background_threshold: float = 0.75,
    ):
        self.train_folder = folder / "train"
        self.test_folder = folder / "test"

        self.train_val_dataset = TreesDataset(
            folder=self.train_folder,
            background_threshold=background_threshold,
        )
        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset,
            [
                int(0.8 * len(self.train_val_dataset)),
                len(self.train_val_dataset) - int(0.8 * len(self.train_val_dataset)),
            ],
        )
        self.train_dataset = DatasetWrapper(
            dataset=self.train_dataset,
            transform=Compose([augmentations, transform]),
        )
        self.val_dataset = DatasetWrapper(
            dataset=self.val_dataset,
            transform=transform,
        )

        self.train = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        self.val = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        self.test_dataset = TreesDataset(
            folder=self.test_folder,
        )
        self.test_dataset = DatasetWrapper(
            dataset=self.test_dataset,
            transform=transform,
        )
        self.test = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        with open(folder / "labels.txt", "r") as f:
            self.labels = {
                line.split(":")[1]: int(line.split(":")[0])
                for line in f.read().splitlines()
            }
            self.num_classes = len(self.labels)


class VOCDataLoaders:
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        transform: Callable = None,
        augmentations: Callable = None,
    ):
        self.num_classes = 21

        def transform_fn(image, mask, to_numpy=True):
            if to_numpy:
                image, mask = np.array(image), np.array(mask)
            transformed = transform(image=image, mask=mask)
            transformed["mask"][transformed["mask"] == 255] = 0
            return transformed["image"], transformed["mask"].type(torch.LongTensor)

        def augmentations_fn(image, mask):
            image, mask = np.array(image), np.array(mask)
            transformed = augmentations(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            return transform_fn(image, mask, to_numpy=False)

        self.train_dataset = VOCSegmentation(
            root="data/VOCdevkit/VOC2012",
            year="2012",
            image_set="train",
            download=False,
        )
        self.train_dataset = DatasetWrapper(
            dataset=self.train_dataset,
            transform=augmentations_fn,
        )
        self.val_dataset = VOCSegmentation(
            root="data/VOCdevkit/VOC2012",
            year="2012",
            image_set="val",
            download=False,
        )
        self.val_dataset = DatasetWrapper(
            dataset=self.val_dataset,
            transform=transform_fn,
        )

        self.train = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        self.val = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
