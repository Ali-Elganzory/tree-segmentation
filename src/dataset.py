import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split


class TreesDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        transform: Callable = None,
    ):
        self.folder = folder
        self.transform = transform
        self.images_folder = folder / "images"
        self.masks_folder = folder / "masks"

        self.data = [Path(p.path) for p in os.scandir(self.images_folder)]
        self.target = list(
            map(lambda x: self.masks_folder / x.name.replace("jpg", "png"), self.data)
        )

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

        image, mask = np.array(image), np.array(mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask


class TreesDataLoaders:
    def __init__(
        self,
        folder: Path,
        batch_size: int,
        num_workers: int = 0,
        transform: Callable = None,
        augmentations: Callable = None,
    ):
        self.train_folder = folder / "train"
        self.test_folder = folder / "test"

        self.train_val_dataset = TreesDataset(
            folder=self.train_folder,
        )
        self.train_dataset, self.val_dataset = random_split(
            self.train_val_dataset,
            [
                int(0.8 * len(self.train_val_dataset)),
                len(self.train_val_dataset) - int(0.8 * len(self.train_val_dataset)),
            ],
        )
        self.train_dataset.dataset.transform = Compose(
            [a for a in [augmentations, transform] if a]
        )
        self.val_dataset.dataset.transform = transform

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
