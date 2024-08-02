from enum import Enum

import torch
import numpy as np
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


class Models(Enum):
    deeplab = "deeplab"
    dino = "dino"

    @property
    def factory(self):
        return {
            Models.deeplab: DeepLabV3Model,
            Models.dino: DinoV2Model,
        }[self]


class DeepLabV3Model(torch.nn.Module):
    transforms = DeepLabV3_ResNet50_Weights.transforms

    augmentations = A.Compose([])

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Change the number of output classes
        self.model.classifier[-1] = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1
        )


class DinoV2Model(torch.nn.Module):
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    transforms = A.Compose(
        [
            A.Resize(504, 504),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
            ToTensorV2(),
        ]
    )

    augmentations = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ]
    )

    def __init__(
        self,
        num_classes: int,
        batch_norm: bool = True,
        skip_batch_norm_on_trans_conv: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # x = torch.Size([32, 384, 36, 36])
        self.head = nn.Sequential(
            # Downsample to 32 x 32
            nn.Conv2d(384, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            # Upsample to 512 x 512
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            (
                nn.BatchNorm2d(64)
                if batch_norm and not skip_batch_norm_on_trans_conv
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            (
                nn.BatchNorm2d(32)
                if batch_norm and not skip_batch_norm_on_trans_conv
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, self.num_classes, kernel_size=4, stride=2, padding=1
            ),
            (
                nn.BatchNorm2d(self.num_classes)
                if batch_norm and skip_batch_norm_on_trans_conv
                else nn.Identity()
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.num_classes, self.num_classes, kernel_size=4, stride=2, padding=1
            ),
            (
                nn.BatchNorm2d(self.num_classes)
                if batch_norm and skip_batch_norm_on_trans_conv
                else nn.Identity()
            ),
            nn.ReLU(),
            # Downsample to 504 x 504
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=5, stride=1),
            nn.BatchNorm2d(self.num_classes) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=5, stride=1),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)["x_norm_patchtokens"]
        x = x.reshape(x.shape[0], 36, 36, self.backbone.embed_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.head(x)
        return x
