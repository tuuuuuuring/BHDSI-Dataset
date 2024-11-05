import os
import cv2
import numpy as np
import rasterio
from alive_progress import alive_bar
from sklearn.metrics import mean_squared_error


# Define the root directory
root_dir = '../data6'

# List of folders to calculate mean and std for
folders_of_interest = ['sentinel1_AREA_vv', 'sentinel1_AREA_vh',
                       'sentinel2_AREA_b2', 'sentinel2_AREA_b3',
                       'sentinel2_AREA_b4', 'sentinel2_AREA_b8']


# Initialize accumulators
mean_accumulator = np.zeros(6)
std_accumulator = np.ones(6)
count = 0  # Total count of pixels

# Traverse each cut directory
with alive_bar(67, force_tty=True) as bar:
    for cut_dir in os.listdir(root_dir):
        cut_path = os.path.join(root_dir, cut_dir)
        if not os.path.isdir(cut_path):
            continue

        # Traverse each folder of interest within the cut directory
        # for i,folder in enumerate(folders_of_interest):
        #     folder_path = os.path.join(cut_path, folder)
        #     if not os.path.exists(folder_path):
        #         continue
        #
        #     # Traverse each TIFF file in the folder
        #     for filename in os.listdir(folder_path):
        #         if filename.endswith('.tif'):
        #             img_path = os.path.join(folder_path, filename)
        #
        #             # Load the image using OpenCV
        #             img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #
        #             # Compute mean and std
        #             img_mean = np.mean(img)
        #
        #             # Update accumulators
        #             mean_accumulator[i] += img_mean
        #             count += 1
        for i,folder in enumerate(folders_of_interest):
            if i<2:
                folder_path = os.path.join(cut_path, 'sentinel1_AREA')
            else:
                folder_path = os.path.join(cut_path, 'sentinel2_AREA')
            if not os.path.exists(folder_path):
                continue

            # Traverse each TIFF file in the folder
            if i<2:
                for filename in os.listdir(folder_path):
                    if filename.endswith('.tif'):
                        img_path = os.path.join(folder_path, filename)
                        with rasterio.open(img_path) as src:
                            img = src.read(i+1)
                        img_mean = np.mean(img)

                        # Update accumulators
                        mean_accumulator[i] += img_mean
            else:
                for filename in os.listdir(folder_path):
                    if filename.endswith('.tif'):
                        img_path = os.path.join(folder_path, filename)
                        with rasterio.open(img_path) as src:
                            img = src.read(i-1)
                        img_mean = np.mean(img)

                        # Update accumulators
                        mean_accumulator[i] += img_mean
        bar()

# Calculate overall mean and std
overall_mean = mean_accumulator / 5606
print(count)
count=5606
means=[]
for i in range(6):
    array = np.full((256, 256), overall_mean[i])
    means.append(array)

with alive_bar(67, force_tty=True) as bar:
    for cut_dir in os.listdir(root_dir):
        cut_path = os.path.join(root_dir, cut_dir)
        if not os.path.isdir(cut_path):
            continue

        # Traverse each TIFF file in the folder
        # for filename in os.listdir(folder_path):
        #     if filename.endswith('.tif'):
        #         img_path = os.path.join(folder_path, filename)
        #
        #         # Load the image using OpenCV
        #         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #
        #         # Compute mean and std
        #         img_std2 = mean_squared_error(img, means[i])
        #
        #         # Update accumulators
        #         std_accumulator[i] += img_std2
        #         count += 1
        for i, folder in enumerate(folders_of_interest):
            if i < 2:
                folder_path = os.path.join(cut_path, 'sentinel1_AREA')
            else:
                folder_path = os.path.join(cut_path, 'sentinel2_AREA')
            if not os.path.exists(folder_path):
                continue

            # Traverse each TIFF file in the folder
            if i < 2:
                for filename in os.listdir(folder_path):
                    if filename.endswith('.tif'):
                        img_path = os.path.join(folder_path, filename)
                        with rasterio.open(img_path) as src:
                            img = src.read(i + 1)

                        # Update accumulators
                        std_accumulator[i] += np.sum((img-overall_mean[i])**2)
            else:
                for filename in os.listdir(folder_path):
                    if filename.endswith('.tif'):
                        img_path = os.path.join(folder_path, filename)
                        with rasterio.open(img_path) as src:
                            img = src.read(i - 1)
                        std_accumulator[i] += np.sum((img-overall_mean[i])**2)
        bar()
overall_std = np.sqrt(std_accumulator / (count*256*256))
print(count)
for i in range(6):
    print(folders_of_interest[i], ":", f"Overall std across all images: {overall_std[i]}")
    print(folders_of_interest[i],":",f"Overall mean across all images: {overall_mean[i]}")