import rasterio
import geopandas as gpd
import numpy as np
import itertools
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.windows import bounds
import os
import fiona
import matplotlib.pyplot as plt
from rasterio.mask import mask
from shapely.geometry import mapping

# Load the .tif file
tif_file = '2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'
annotations_file = 'Z1_polygons.gpkg'
test_polygons_file = 'inference_zone.gpkg'
train_output_dir = 'train_data'
test_output_dir = 'test_data'

# Load the annotations and test polygons
annotations = gpd.read_file(annotations_file)
test_polygons = gpd.read_file(test_polygons_file)

# Create output directories
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Define tile size and chunk size
tile_size = 256
chunk_size = 4096  # Adjust this based on your memory constraints

# Define a function to process a window
def process_window(window, row_offset, col_offset, src, profile, annotations, test_polygons, train_output_dir, test_output_dir):
    # Read the window data
    tile = src.read(window=window)
    
    # Create bounding box for the tile
    bbox = box(*bounds(window, src.transform))
    
    # Check intersection with test polygons
    is_test_tile = test_polygons.intersects(bbox).any()
    # print(type(is_test_tile))
    # print(is_test_tile)
    # Define the output directory based on the check
    output_dir = test_output_dir if is_test_tile else train_output_dir
    
    # Check intersection with annotations
    intersecting_annotations = annotations.intersects(bbox)
    
    # Save the mask
    mask = annotations[intersecting_annotations]
    if intersecting_annotations.any():
        # tile_filename = os.path.join(output_dir, f'tile_{row_offset}_{col_offset}.tif')
        tile_filename = os.path.join(output_dir, f'tile_test.tif')
        with rasterio.open(tile_filename, 'w', **profile) as dst:
            dst.write(tile)
        print(mask)
        mask_filename = os.path.join(output_dir, f'mask_test.geojson')
        mask.to_file(mask_filename, driver='GeoJSON')

        # with fiona.open(f'{output_dir}/mask_test.geojson', "r") as geojson:
        #     features = [feature["geometry"] for feature in geojson]
        # with rasterio.open('2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif') as src:
        #     out_image, _ = rasterio.mask.mask(src, features, crop=True)
        # with rasterio.open(f'{output_dir}/tile_test.tif') as src:
        #     out_image_1, _ = rasterio.mask.mask(src, features, crop=True)
        # plt.imshow(out_image_1.transpose(1, 2, 0))
        # plt.title(f'out image 1')
        # plt.show()
        # plt.imshow(out_image.transpose(1, 2, 0))
        # plt.title(f'out image')
        # plt.show()
        # exit()
        # out_image, out_transform = rasterio.mask.mask(tile, features, crop=True)
        #show out_image using matplotlib
        # if out_image.shape[0] >= 3:
        #     # Combine the first three bands into an RGB image
        #     rgb_image = np.dstack((out_image[0], out_image[1], out_image[2]))
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(rgb_image)
        #     plt.title('RGB Composite Image')
        #     plt.axis('off')
        #     plt.show()
        #     exit()
        # else:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(out_image[0], cmap='gray')  # Assuming a single band for simplicity
        #     plt.title('Masked Image')
        #     plt.axis('off')
        #     plt.show()

# Open the .tif file and process in chunks
with rasterio.open(tif_file) as src:
    profile = src.profile
    
    # Calculate number of chunks
    nrows = src.height // chunk_size + (src.height % chunk_size > 0)
    ncols = src.width // chunk_size + (src.width % chunk_size > 0)
    
    for i in range(nrows):
        for j in range(ncols):
            # Define the window
            window = Window(j * chunk_size, i * chunk_size, chunk_size, chunk_size).intersection(Window(0, 0, src.width, src.height))
            
            # Process the window in smaller tiles
            nrows_tile = window.height // tile_size + (window.height % tile_size > 0)
            ncols_tile = window.width // tile_size + (window.width % tile_size > 0)
            
            for row in range(nrows_tile):
                for col in range(ncols_tile):
                    tile_window = Window(window.col_off + col * tile_size, window.row_off + row * tile_size, tile_size, tile_size).intersection(window)
                    
                    if tile_window.width == tile_size and tile_window.height == tile_size:
                        process_window(tile_window, i * nrows_tile + row, j * ncols_tile + col, src, profile, annotations, test_polygons, train_output_dir, test_output_dir)