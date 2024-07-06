import geopandas as gpd
import fiona
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np

gpkg_file_path = 'Z1_polygons.gpkg'
# gpkg_file_path = 'inference_zone.gpkg'
# # gpkg_file_path = 'Z2_polygons.gpkg'
# # gpkg_file_path = 'Z3_polygons.gpkg'
tif_file_path = '2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'
# tif_file_path = "train_data/tile_test.tif"
# # Open the GeoPackage and list all layer names
# with fiona.Env():
#     with fiona.open(gpkg_file_path, layer=None) as src:
#         layer_names = fiona.listlayers(gpkg_file_path)
#         print("Layer names in the GeoPackage:")
#         for layer in layer_names:
#             print(layer)



# Open the raster file
with rasterio.open(tif_file_path) as src:
    # Define the window (subset) you want to read and plot
    window = rasterio.windows.Window(col_off=20000, row_off=20000, width=1024, height=1024)  # Adjust dimensions as needed

    # Read the subset of the raster data
    subset = src.read(window=window)
    subset_transform = src.window_transform(window)
    subset_crs = src.crs

# Read polygons from GeoPackage
gdf = gpd.read_file(gpkg_file_path)  # Replace with actual layer name

# Ensure the polygons are within the bounds of the raster subset
# bbox = box(*src.bounds)  # Create a bounding box of the raster subset
# gdf = gdf[gdf.geometry.intersects(bbox)]  # Filter polygons that intersect with the subset bounding box

# Plot the subset of the raster data
plt.figure(figsize=(10, 10))
show(subset, transform=subset_transform, cmap='gray')
plt.title('Subset of Raster Data')

# Overlay the polygons as segmentation boundaries
# for idx, row in gdf.iterrows():
#     boundary = row['geometry']
#     gpd.GeoSeries([boundary]).plot(ax=plt.gca(), facecolor='none', edgecolor='red', linewidth=2)
# print("Number of polygons:", len(gdf))
# plt.show()

# # Function to extract raster data for each polygon
def extract_raster_data(polygon, src):
    # Convert polygon to GeoJSON format
    print("polygon: ", polygon)
    geojson_polygon = [mapping(polygon)]
    print("geojson_polygon: ", geojson_polygon)
    # Mask the raster with the polygon
    out_image, out_transform = mask(src, geojson_polygon, crop=True)
    
    return out_image, out_transform

# columns = gdf.columns
# print("columns: ", columns)

# # Open the raster file
with rasterio.open(tif_file_path) as src:
    for idx, row in gdf.iterrows():
        polygon = row['geometry']
        print("type(polygon): ", type(polygon))
        out_image, out_transform = extract_raster_data(polygon, src)
        
        # Plot the extracted raster data
        plt.imshow(out_image.transpose(1, 2, 0))
        plt.title(f'Extracted Raster Data for Polygon {idx}')
        plt.show()

