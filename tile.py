import os

import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from rasterio.io import DatasetReader
from rasterio.windows import Window, bounds
from rasterio.mask import raster_geometry_mask


DATASET_FOLDER = "data/quebec_trees_dataset_2021-09-02/"
ZONE_NUM = 3
ZONE1_TIF_FILEPATH_TEMPLATE = (
    DATASET_FOLDER + "2021-09-02/zone{}/2021-09-02-sbl-z{}-rgb-cog.tif"
)
ANNOTATION_FILEPATH_TEMPLATE = DATASET_FOLDER + "Z{}_polygons.gpkg"
TEST_POLYGONS_FILEPATH = DATASET_FOLDER + "inference_zone.gpkg"
TEST_POLYGONS = gpd.read_file(TEST_POLYGONS_FILEPATH)

TRAIN_SPLIT_DIR = DATASET_FOLDER + "train"
TEST_SPLIT_DIR = DATASET_FOLDER + "test"

TILE_SIZE = 504
LABELS: dict[str, int] = {
    "Background": 0,
}

MAP: dict[str, str] = {
    "Acer": "ACPE",
    "PIGL": "Picea",
    "PIMA": "Picea",
    "PIRU": "Picea",
    "POGR": "Populus",
    "POTR": "Populus",
    "QURU": "BEAL",
    "Feuillus": "ACSA",
    "Conifere": "Picea",
    "FRNI": "BEAL",
    "Betula": "BEAL",
    "PRPE": "ACSA",
    "POBA": "Populus",
    "BEPO": "BEPA",
    "OSVI": "FAGR",
    # "QURU": "Background",
    # "Feuillus": "Background",
    # "Conifere": "Background",
    # "FRNI": "Background",
    # "Betula": "Background",
    # "PRPE": "Background",
    # "POBA": "Background",
    # "BEPO": "Background",
    # "OSVI": "Background",
}


def save_tile(
    tile: np.ndarray,
    path: str,
):
    # Only save if not all black
    if np.any(tile):
        Image.fromarray(tile, mode="RGB" if tile.shape[-1] == 3 else "L").save(path)


def process_window(
    window: Window,
    src: DatasetReader,
    path: str,
):
    tile = src.read(window=window)
    tile = tile[:3].transpose(1, 2, 0)

    save_tile(tile, path)


def is_test_tile(window: Window) -> bool:
    return TEST_POLYGONS.intersects(
        box(
            *bounds(
                window,
                src.transform,
            )
        )
    ).any()


if __name__ == "__main__":
    for zone in range(1, 4):
        print(f"Processing zone {zone}\n")

        os.makedirs(TRAIN_SPLIT_DIR + "/images", exist_ok=True)
        os.makedirs(TRAIN_SPLIT_DIR + "/masks", exist_ok=True)
        os.makedirs(TEST_SPLIT_DIR + "/images", exist_ok=True)
        os.makedirs(TEST_SPLIT_DIR + "/masks", exist_ok=True)

        src: DatasetReader
        with rasterio.open(ZONE1_TIF_FILEPATH_TEMPLATE.format(zone, zone)) as src:
            profile = src.profile
            nrows = src.height // TILE_SIZE
            ncols = src.width // TILE_SIZE

            # Images
            for row in tqdm(range(nrows), desc="Processing images"):
                for col in range(ncols):
                    window = Window(
                        col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE
                    )

                    filename = os.path.join(
                        (
                            (
                                TRAIN_SPLIT_DIR
                                if not is_test_tile(window)
                                else TEST_SPLIT_DIR
                            )
                            + "/images"
                        ),
                        f"tile_{window.row_off // TILE_SIZE}_{window.col_off // TILE_SIZE}.jpg",
                    )
                    process_window(window, src, filename)

            # Masks (annotations)
            annotations: gpd.GeoDataFrame = gpd.read_file(
                ANNOTATION_FILEPATH_TEMPLATE.format(zone)
            )
            annotations["Label"] = annotations["Label"].replace(MAP)
            unique_labels = annotations["Label"].unique()
            for label in unique_labels:
                if label not in LABELS:
                    LABELS[label] = len(LABELS)
            annotations["Label"] = annotations["Label"].map(LABELS).astype(int)

            mask = np.zeros((src.height, src.width), dtype=np.uint8)
            for label in tqdm(unique_labels, desc="Processing labels"):
                label_annotations = annotations[annotations["Label"] == LABELS[label]]
                label_mask, _, _ = raster_geometry_mask(
                    src,
                    label_annotations["geometry"],
                    invert=True,
                )
                mask[label_mask] = LABELS[label]

            for row in tqdm(range(nrows), desc="Processing masks"):
                for col in range(ncols):
                    window = Window(
                        col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE
                    )

                    filename = os.path.join(
                        (
                            TRAIN_SPLIT_DIR
                            if not is_test_tile(window)
                            else TEST_SPLIT_DIR
                        )
                        + "/masks",
                        f"tile_{row}_{col}.png",
                    )

                    save_tile(
                        mask[
                            row * TILE_SIZE : (row + 1) * TILE_SIZE,
                            col * TILE_SIZE : (col + 1) * TILE_SIZE,
                        ],
                        filename,
                    )

            print("\n" + ("-" * 50) + "\n")

        # Save labels
        with open(DATASET_FOLDER + "labels.txt", "w") as f:
            f.write("\n".join(f"{label}: {name}" for name, label in LABELS.items()))
