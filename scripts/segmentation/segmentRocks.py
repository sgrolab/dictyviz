import zarr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal
GAUSSIAN_FILTER_SIZE = 1.5  # standard deviation of the Gaussian filter kernel applied to the rocks channel

def removeCellShadows(resArray, channels, threshold):
    """    Remove shadows of cells from the rocks image by applying a median filter
    to the pixels in the rocks image that are shadowed by cells.
    """
    rocks = resArray[:, channels["rocks"]]
    cells = resArray[:, channels["cells"]]

    rocksImgFiltered = rocks.copy()
    cellsImgMasked = cv2.merge([cells, cells, cells]).astype("uint8")  # Create a 3-channel mask from the single channel cells image

    for t in tqdm(range(rocks.shape[0]), desc="Removing cell shadows"):
        for i, j, k in np.ndindex(rocks.shape[1:]):
            # Check if the pixel value is above the threshold
            if cells[t, k, i, j] > threshold:
                # Apply a median filter to the corresponding rocks pixel
                localMedian = np.median(rocks[t, max(0,i-SHADOW_MED_FILT):min(rocks.shape[1],i+SHADOW_MED_FILT), max(0,j-SHADOW_MED_FILT):min(rocks.shape[2],j+SHADOW_MED_FILT)])
                rocksImgFiltered[t, k, i, j] = localMedian
                cellsImgMasked[t, k, i, j, :] = [255, 0, 0]
    return rocksImgFiltered, cellsImgMasked

def __main__():
    # Load the Zarr dataset
    zarrPath = 'path/to/your/zarr/dataset.zarr'
    root = zarr.open(zarrPath, mode='r')
    resArray = root['0']['0'] 

    # set channels
    channels = {"cells": 0, "rocks": 1}

    threshold = np.percentile(resArray[0, channels["cells"], 5, :, :], THRESHOLD)

    # remove shadows from the rocks image
    rocksImgFiltered, cellsImgMasked = removeCellShadows(resArray, channels, threshold)
