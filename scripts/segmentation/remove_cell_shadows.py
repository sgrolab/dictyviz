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

from dictyviz.utils import get_axes, get_channels

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal


def process_timepoint_gpu(volume, threshold, device='cuda'):
    """
    GPU-optimized processing of a single timepoint across all z-slices.
    """

    # Convert to native byte order before PyTorch conversion
    rocks_volume = np.ascontiguousarray(volume[0], dtype=np.float32)
    cells_volume = np.ascontiguousarray(volume[1], dtype=np.float32)

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

def remove_cell_shadows(rocks_img, cells_img, threshold):
    """    Remove shadows of cells from the rocks image by applying a median filter
    to the pixels in the rocks image that are shadowed by cells.
    """
    rocks_img_filtered = rocks_img.copy()
    cells_img_masked = cv2.merge([cells_img, cells_img, cells_img]).astype("uint8")  # Create a 3-channel mask from the single channel cells image
    for i in range(cells_img.shape[0]):
        for j in range(cells_img.shape[1]):
            # Check if the pixel value is above the threshold
            if cells_img[i, j] > threshold:
                #apply a median filter to the corresponding rocks pixel
                local_median = np.median(rocks_img[max(0,i-SHADOW_MED_FILT):min(rocks_img.shape[0],i+SHADOW_MED_FILT), max(0,j-SHADOW_MED_FILT):min(rocks_img.shape[1],j+SHADOW_MED_FILT)])
                rocks_img_filtered[i, j] = local_median
                cells_img_masked[i, j, :] =  [255, 0, 0]  # Mark the cell shadow in red for visualization
    return rocks_img_filtered, cells_img_masked

def __main__():
    #TODO: switch to saving as zarrs instead of tiffs

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        print("No GPU available. Exiting.")
        exit(1)

    # Load the Zarr dataset
    zarr_path = sys.argv[1]
    print(f"Loading Zarr dataset from {zarr_path}...")
    parent_dir = os.path.dirname(zarr_path) + "/"
    output_zarr = parent_dir + "segmentation/"
    if not os.path.exists(output_zarr):
        os.makedirs(output_zarr)

    root = zarr.open(zarr_path, mode='r')
    res_array = root['0']['0']

    # Set z range from command line arguments
    z_range = range(int(sys.argv[2]), int(sys.argv[3]))
    Z = len(z_range)

    T, C, _, Y, X = res_array.shape

    output_root = zarr.open(output_zarr, mode='a')
    output_root.create_dataset(
        "shadows_removed", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint16', overwrite=True
    )

    axes = get_axes(zarr_path)
    output_root["shadows_removed"].attrs['axes'] = axes
    output_root["shadows_removed"].attrs['shadow_removal_threshold_percentile'] = THRESHOLD
    output_root["shadows_removed"].attrs['shadow_removal_median_filter_size'] = SHADOW_MED_FILT

    # set channels
    channels = get_channels(zarr_path + '/parameters.json')
    cells = next((i for i, ch in enumerate(channels) if ch.name == 'cells'), None)
    rocks = next((i for i, ch in enumerate(channels) if ch.name == 'rocks'), None)

    if cells is None or rocks is None:
        # default to channels 0 and 1
        print("Warning: 'cells' or 'rocks' channel not found in parameters.json. Defaulting to channels 0 and 1.")
        cells = 0
        rocks = 1

    # could be recalculated for each time point, but for now we use the first time point
    threshold = np.percentile(res_array[0, cells, 5, :, :], THRESHOLD)

    for t in range(T):

        # tpOutputDir = outputDir + f"t{t}/"
        # if not os.path.exists(tpOutputDir):
        #     os.makedirs(tpOutputDir)

        # rocksImgPath = tpOutputDir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
        #cellsImgPath = tpOutputDir + "cell_mask_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
        if torch.cuda.is_available():
            try:
                # Get all z-slices for this timepoint
                volume = res_array[t, :, z_range, :, :]  # Shape: (Z, H, W)
                # TODO: feed channel info instead of hardcoding
                rocks_img_filtered, masked_cells = process_timepoint_gpu(
                    volume, threshold, device
                )

                rocks_img_filtered = rocks_img_filtered.astype(np.uint16)
                output_root["shadows_removed"][t] = rocks_img_filtered
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print("Error processing with GPU")
                import traceback
                traceback.print_exc()
                # finalImg = np.zeros((resArray.shape[2], resArray.shape[3], resArray.shape[4]), dtype=np.uint16)  # Initialize the array for filtered rocks image
                # maskedCells = []  # Create a 3-channel mask from the single channel cells image

                # for zSlice in tqdm(range(resArray.shape[2]), desc="Removing cell shadows"):
                #     rocksImg = resArray[t, rocks, zSlice, :, :]
                #     cellsImg = resArray[t, cells, zSlice, :, :]

                #     rocksImgFiltered, cellsImgMasked = removeCellShadows(rocksImg, cellsImg, threshold)

                #     finalImg[zSlice, :, :] = rocksImgFiltered
                #     maskedCells.append(cellsImgMasked)
                # rocksImgFiltered = finalImg
                # maskedCells = np.array(maskedCells)
        else:
            print("GPU not available")
            # finalImg = np.zeros((resArray.shape[2], resArray.shape[3], resArray.shape[4]), dtype=np.uint16)  # Initialize the array for filtered rocks image
            # maskedCells = []
            # for zSlice in tqdm(range(resArray.shape[2]), desc="Removing cell shadows"):

            #     rocksImg = resArray[t, rocks, zSlice, :, :]
            #     cellsImg = resArray[t, cells, zSlice, :, :]

            #     rocksImgFiltered, cellsImgMasked = removeCellShadows(rocksImg, cellsImg, threshold)

            #     finalImg[zSlice, :, :] = rocksImgFiltered
            #     maskedCells.append(cellsImgMasked)
            # rocksImgFiltered = finalImg
            # maskedCells = np.array(maskedCells)

        # clear memory
        del rocks_img_filtered
        del masked_cells

    print("Cell shadow removal complete. Results saved to ", output_zarr)

if __name__ == "__main__":
    __main__()
