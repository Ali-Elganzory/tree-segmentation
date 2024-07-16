import os
from pathlib import Path

from PIL import Image
from typer import Typer
import matplotlib.pyplot as plt

app = Typer()


@app.command()
def mask(
    path: str,
):
    os.makedirs("./visualization", exist_ok=True)

    mask_filepath = Path(path)
    image_filepath = (
        mask_filepath.parent.parent
        / "images"
        / mask_filepath.name.replace("png", "jpg")
    )

    mask = Image.open(mask_filepath)
    image = Image.open(image_filepath)

    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(mask)
    ax[1].set_title("Mask")
    ax[1].axis("off")
    ax[2].imshow(image)
    ax[2].imshow(mask, alpha=0.3)
    ax[2].set_title("Overlay")
    ax[2].axis("off")
    plt.savefig(f"./visualization/{mask_filepath.name}")


if __name__ == "__main__":
    app()
