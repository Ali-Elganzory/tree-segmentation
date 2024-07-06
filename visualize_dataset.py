import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Open the dataset
tif_file_path = '2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'
# tif_file_path = '2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif'
# tif_file_path = 'train_data/tile_test.tif'
# tif_file_path = '2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif'

row_start, row_end = 24000, 24100  # Example row range
col_start, col_end = 24000, 24100  # Example column range

# with rasterio.open(tif_file_path) as src:
#     # Read the specified window from each band
#     window = ((row_start, row_end), (col_start, col_end))
#     data_subset = src.read(window=window)

#     # If it's RGB data and has at least 3 bands
#     if data_subset.shape[0] >= 3:
#         rgb = np.dstack((data_subset[0], data_subset[1], data_subset[2]))
#         plt.imshow(rgb)
#         plt.title('RGB Composite of Subset')
#         plt.show()
#     else:
#         plt.imshow(data_subset[0], cmap='gray')
#         plt.colorbar()
#         plt.title('Subset of Band 1')
#         plt.show()

# for i in range(0, 15):
#     for j in range(0, 16):
#         with rasterio.open(f'train_data/tile_{i}_{j}.tif') as src:
#             data_overview = src.read(out_shape=(src.count, src.height // 10, src.width // 10))
    
#             # Adjust the transform to match the new shape
#             transform = src.transform * src.transform.scale((src.width / data_overview.shape[-1]), (src.height / data_overview.shape[-2]))

#             # Display the overview
#             if data_overview.shape[0] >= 3:
#                 rgb = np.dstack((data_overview[0], data_overview[1], data_overview[2]))
#                 plt.imshow(rgb)
#                 plt.title('RGB Composite Overview')
#                 plt.show()
#             else:
#                 plt.imshow(data_overview[0], cmap='gray')
#                 plt.colorbar()
#                 plt.title('Grayscale Overview')
#                 plt.show()


with rasterio.open(tif_file_path) as src:
    # Read data with reduced resolution
    data_overview = src.read(out_shape=(src.count, src.height // 10, src.width // 10))
    
    # Adjust the transform to match the new shape
    transform = src.transform * src.transform.scale((src.width / data_overview.shape[-1]), (src.height / data_overview.shape[-2]))

    # Display the overview
    if data_overview.shape[0] >= 3:
        rgb = np.dstack((data_overview[0], data_overview[1], data_overview[2]))
        plt.imshow(rgb)
        plt.title('RGB Composite Overview')
        plt.show()
    else:
        plt.imshow(data_overview[0], cmap='gray')
        plt.colorbar()
        plt.title('Grayscale Overview')
        plt.show()