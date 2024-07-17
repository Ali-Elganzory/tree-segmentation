import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DinoV2Model(torch.nn.Module):
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    transforms = A.Compose(
        [
            A.Resize(224, 224),
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

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True
        )
        # Upscale to 224x224, head with multiple layers
        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=self.backbone.embed_dim,
                out_channels=self.num_classes,
                kernel_size=14,
                stride=14,
            ),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)["x_norm_patchtokens"]
        x = x.reshape(x.shape[0], 16, 16, self.backbone.embed_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.head(x)
        return x
