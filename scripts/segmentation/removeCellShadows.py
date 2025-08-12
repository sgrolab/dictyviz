import os
import sys
import zarr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

import torch
import torch.nn.functional as F

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal

def process_timepoint_gpu(resArray, t, rocks_ch, cells_ch, threshold, device='cuda'):
    """
    GPU-optimized processing of a single timepoint across all z-slices.
    """
    # Get all z-slices for this timepoint
    rocks_volume = resArray[t, rocks_ch, :, :, :]  # Shape: (Z, H, W)
    cells_volume = resArray[t, cells_ch, :, :, :]  # Shape: (Z, H, W)

    # Convert to native byte order before PyTorch conversion
    rocks_volume = np.ascontiguousarray(rocks_volume, dtype=np.float32)
    cells_volume = np.ascontiguousarray(cells_volume, dtype=np.float32)    
    
    # Convert to torch tensors and move to GPU
    rocks_tensor = torch.from_numpy(rocks_volume).to(device)
    cells_tensor = torch.from_numpy(cells_volume).to(device)
    
    # Batch process all slices at once
    Z, H, W = rocks_tensor.shape
    
    # Create shadow mask for all slices
    shadow_mask = cells_tensor > threshold
    
    # Apply 2D median filter to each slice using convolution
    kernel_size = SHADOW_MED_FILT * 2 + 1
    padding = SHADOW_MED_FILT
    
    # Reshape for batch processing: (Z, 1, H, W)
    rocks_batch = rocks_tensor.unsqueeze(1)
    
    # Pad the volume
    rocks_padded = F.pad(rocks_batch, (padding, padding, padding, padding), mode='reflect')
    
    # Create sliding windows for median calculation
    windows = rocks_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # Shape: (Z, 1, H, W, kernel_size, kernel_size)
    
    # Flatten windows and calculate median
    windows_flat = windows.reshape(Z, H, W, -1)
    local_medians = torch.median(windows_flat, dim=3)[0]
    
    # Apply shadow correction
    rocks_filtered = rocks_tensor.clone()
    rocks_filtered[shadow_mask] = local_medians[shadow_mask]
    
    # Create visualization masks
    cells_masked = torch.stack([cells_tensor, cells_tensor, cells_tensor], dim=3)
    cells_masked[shadow_mask] = torch.tensor([255, 0, 0], dtype=torch.float, device=device)
    
    return rocks_filtered.cpu().numpy(), cells_masked.cpu().numpy().astype(np.uint8)

def removeCellShadows(rocksImg, cellsImg, threshold):
    """    Remove shadows of cells from the rocks image by applying a median filter
    to the pixels in the rocks image that are shadowed by cells.
    """
    rocksImgFiltered = rocksImg.copy()
    cellsImgMasked = cv2.merge([cellsImg, cellsImg, cellsImg]).astype("uint8")  # Create a 3-channel mask from the single channel cells image
    for i in range(cellsImg.shape[0]):
        for j in range(cellsImg.shape[1]):
            # Check if the pixel value is above the threshold
            if cellsImg[i, j] > threshold:
                #apply a median filter to the corresponding rocks pixel
                localMedian = np.median(rocksImg[max(0,i-SHADOW_MED_FILT):min(rocksImg.shape[0],i+SHADOW_MED_FILT), max(0,j-SHADOW_MED_FILT):min(rocksImg.shape[1],j+SHADOW_MED_FILT)])
                rocksImgFiltered[i, j] = localMedian
                cellsImgMasked[i, j, :] =  [255, 0, 0]  # Mark the cell shadow in red for visualization
    return rocksImgFiltered, cellsImgMasked

def __main__():
    #TODO: switch to saving as zarrs instead of tiffs

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")

    # Load the Zarr dataset
    zarrPath = sys.argv[1]
    print(f"Loading Zarr dataset from {zarrPath}...")
    parentDir = os.path.dirname(zarrPath) + "/"
    outputDir = parentDir + "segmentation/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    root = zarr.open(zarrPath, mode='r')
    resArray = root['0']['0'] 

    # set channels
    cells = 0
    rocks = 1

    # Set time range from command line arguments
    tRange = range(int(sys.argv[2]), int(sys.argv[3]))

    # could be recalculated for each time point, but for now we use the first time point
    threshold = np.percentile(resArray[0, cells, 5, :, :], THRESHOLD)

    for t in tRange:

        tpOutputDir = outputDir + f"t{t}/"
        if not os.path.exists(tpOutputDir):
            os.makedirs(tpOutputDir)

        rocksImgPath = tpOutputDir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
        #cellsImgPath = tpOutputDir + "cell_mask_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
        if not os.path.exists(rocksImgPath):
            print(f"\nRemoving cell shadows from the rocks image for timepoint {t}...")

            if torch.cuda.is_available():
                try:
                    rocksImgFiltered, maskedCells = process_timepoint_gpu(
                        resArray, t, rocks, cells, threshold, device
                    )
                    
                    rocksImgFiltered = rocksImgFiltered.astype(np.uint16)
                    
                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print("Error processing with GPU. Falling back to CPU processing.")
                    print(f"Error details: {e}")
                    import traceback
                    traceback.print_exc()
                    finalImg = np.zeros((resArray.shape[2], resArray.shape[3], resArray.shape[4]), dtype=np.uint16)  # Initialize the array for filtered rocks image
                    maskedCells = []  # Create a 3-channel mask from the single channel cells image

                    for zSlice in tqdm(range(resArray.shape[2]), desc="Removing cell shadows"):
                        rocksImg = resArray[t, rocks, zSlice, :, :]
                        cellsImg = resArray[t, cells, zSlice, :, :]

                        rocksImgFiltered, cellsImgMasked = removeCellShadows(rocksImg, cellsImg, threshold)

                        finalImg[zSlice, :, :] = rocksImgFiltered
                        maskedCells.append(cellsImgMasked)
            else:
                for zSlice in tqdm(range(resArray.shape[2]), desc="Removing cell shadows"):
                    finalImg = np.zeros((resArray.shape[2],resArray.shape[3], resArray.shape[4]), dtype=np.uint16)  # Initialize the array for filtered rocks image
                    maskedCells = []

                    rocksImg = resArray[t, rocks, zSlice, :, :]
                    cellsImg = resArray[t, cells, zSlice, :, :]

                    rocksImgFiltered, cellsImgMasked = removeCellShadows(rocksImg, cellsImg, threshold)

                    finalImg[zSlice, :, :] = rocksImgFiltered
                    maskedCells.append(cellsImgMasked)
                rocksImgFiltered = finalImg
                maskedCells = np.array(maskedCells)

            tiff.imwrite(rocksImgPath, rocksImgFiltered.astype("uint16"))
            #tiff.imwrite(cellsImgPath, maskedCells.astype("uint16"))
            print(f"Cell shadows removed and saved to {tpOutputDir}")

if __name__ == "__main__":
    __main__()
