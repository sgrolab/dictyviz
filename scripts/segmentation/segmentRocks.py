import os
import sys
import zarr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal
GAUSSIAN_FILTER_SIZE = 2.2  # standard deviation of the Gaussian filter kernel applied to the rocks channel

def removeCellShadows(resArray, channels, threshold):
    """    Remove shadows of cells from the rocks image by applying a median filter
    to the pixels in the rocks image that are shadowed by cells.
    """
    rocks = resArray[:, channels["rocks"]]
    cells = resArray[:, channels["cells"]]

    rocksImgFiltered = rocks.copy()
    cellsImgMasked = cv2.merge([cells, cells, cells]).astype("uint8")  # Create a 3-channel mask from the single channel cells image

    # Set short t range for testing
    tRange = range(0, 5)

    for t in tqdm(tRange, desc="Removing cell shadows"):
        for i, j, k in np.ndindex(rocks.shape[1:]):
            # Check if the pixel value is above the threshold
            if cells[t, k, i, j] > threshold:
                # Apply a median filter to the corresponding rocks pixel
                localMedian = np.median(rocks[t, max(0,i-SHADOW_MED_FILT):min(rocks.shape[1],i+SHADOW_MED_FILT), max(0,j-SHADOW_MED_FILT):min(rocks.shape[2],j+SHADOW_MED_FILT)])
                rocksImgFiltered[t, k, i, j] = localMedian
                cellsImgMasked[t, k, i, j, :] = [255, 0, 0]
    return rocksImgFiltered, cellsImgMasked

def applyGaussianFilter(rocksImgFiltered):
    """Apply a Gaussian filter to the rocks image to smooth it."""
    rocksGaussianFiltered = np.zeros_like(rocksImgFiltered)  # Initialize the array for OpenCV filtered image

    for zSlice in tqdm(range(rocksImgFiltered.shape[0])):
        rocksGaussianFiltered[zSlice] = cv2.GaussianBlur(rocksImgFiltered[zSlice], (0, 0), sigmaX=GAUSSIAN_FILTER_SIZE, sigmaY=GAUSSIAN_FILTER_SIZE)

    return rocksGaussianFiltered

def applyOtsuThreshold(rocksGaussianFiltered):
    """Apply Otsu's thresholding to the Gaussian filtered rocks image."""
    from skimage.filters import threshold_otsu

    otsuThreshold = threshold_otsu(rocksGaussianFiltered)
    rocksOtsuThreshold = rocksGaussianFiltered > otsuThreshold

    return rocksOtsuThreshold

def segmentRocksWatershed(rocksOtsuThreshold, zRange):
    """Segment rocks using the watershed algorithm."""

    #TODO: come up with an automated way to determine the zRange

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    middleSlice = (zRange[0] + zRange[1]) // 2

    # Generate markers from the center zSlice as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(rocksOtsuThreshold[zRange[0]:zRange[1]])
    distance_smoothed = ndi.gaussian_filter(distance, sigma=21)
    coords = peak_local_max(distance_smoothed[middleSlice], 
                            min_distance=50, 
                            threshold_rel=0.1, 
                            exclude_border=False, 
                            footprint=np.ones((50, 50)))
    print(f"Found {len(coords)} local maxima as markers.")

    # Use the markers to fill out the watershed in all zSlices
    mask = np.zeros(rocksOtsuThreshold.shape, dtype=bool)
    mask[zRange[0]:zRange[1]][tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=rocksOtsuThreshold[zRange[0]:zRange[1]])

    return coords, labels

def __main__():
    # Load the Zarr dataset
    zarrPath = sys.argv[1]
    print(f"Loading Zarr dataset from {zarrPath}...")
    parentDir = os.path.dirname(zarrPath) + "/"
    outputDir = parentDir + "segmentation/"
    root = zarr.open(zarrPath, mode='r')
    resArray = root['0']['0'] 

    # set channels
    channels = {"cells": 0, "rocks": 1}

    rocksImgPath = outputDir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
    cellsImgPath = outputDir + "cell_mask_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
    if not os.path.exists(rocksImgPath) or not os.path.exists(cellsImgPath):
        print("Removing cell shadows from the rocks image...")
        threshold = np.percentile(resArray[0, channels["cells"], 5, :, :], THRESHOLD)

        # remove shadows from the rocks image
        rocksImgFiltered, cellsImgMasked = removeCellShadows(resArray, channels, threshold)

        tiff.imwrite(rocksImgPath, rocksImgFiltered.astype("uint16"))
        tiff.imwrite(cellsImgPath, cellsImgMasked.astype("uint16"))
        print(f"Cell shadows removed and saved to {outputDir}")
    else:
        print(f"Cell shadows already removed. Skipping processing for {rocksImgPath} and {cellsImgPath}")
        rocksImgFiltered = tiff.imread(rocksImgPath)
        cellsImgMasked = tiff.imread(cellsImgPath)

    rocksOtsuPath = outputDir + "rocks_gaussian_filtered_otsu_threshold.tif"
    if not os.path.exists(rocksOtsuPath):
        print("Applying Gaussian filter and Otsu's thresholding to the rocks image...")
        # Invert rock channel
        rocksImgFiltered = 65535 - rocksImgFiltered

        rocksGaussianFiltered = np.zeros_like(rocksImgFiltered)  # Initialize the array for OpenCV filtered image
        rocksOtsuThreshold = np.zeros_like(rocksImgFiltered, dtype=bool)

        for t in tqdm(range(rocksImgFiltered.shape[0]), desc="Applying Gaussian filter"):
            # Apply Gaussian filter to the rocks image
            rocksGaussianFiltered[t] = applyGaussianFilter(rocksImgFiltered[t])

            # Apply Otsu's thresholding to the Gaussian filtered rocks image
            rocksOtsuThreshold[t] = applyOtsuThreshold(rocksGaussianFiltered[t])

        #Save the Otsu thresholded image
        tiff.imwrite(rocksOtsuPath, rocksOtsuThreshold.astype("uint16"))
        print(f"Otsu thresholded rocks image saved to {rocksOtsuPath}")
    else:
        print(f"Otsu thresholded rocks image already exists. Skipping processing for {rocksOtsuPath}")
        rocksOtsuThreshold = tiff.imread(rocksOtsuPath)

    segmentedRocksPath = outputDir + "segmented_rocks.tif"
    if not os.path.exists(segmentedRocksPath):
        print("Segmenting rocks using watershed...")

        # Define the zRange for segmentation
        zRange = (4, 100) # Adjust this range as needed
    
        coords = []
        labels = np.zeros_like(rocksOtsuThreshold, dtype=np.uint16)
        for t in tqdm(range(rocksOtsuThreshold.shape[0]), desc="Segmenting rocks"):
            coords_t, labels_t = segmentRocksWatershed(rocksOtsuThreshold[t], zRange)
            coords.append(coords_t)
            labels[t] = labels_t

        # Save the segmented rocks image
        tiff.imwrite(segmentedRocksPath, labels.astype("uint16"))
        print(f"Segmented rocks image saved to {segmentedRocksPath}")
    else:
        print(f"Segmented rocks image already exists. Skipping processing for {segmentedRocksPath}")
        labels = tiff.imread(segmentedRocksPath)

    # Calculate the centroids of the segmented rocks over time

if __name__ == "__main__":
    __main__()

