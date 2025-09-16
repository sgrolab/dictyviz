import os
import sys
import zarr
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import networkx as nx

from scipy import ndimage as ndi
<<<<<<< HEAD
=======
from skimage.filters import threshold_mean
>>>>>>> main
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from laptrack import LapTrack
from laptrack.data_conversion import convert_split_merge_df_to_napari_graph

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal
GAUSSIAN_FILTER_SIZE = 2.2  # standard deviation of the Gaussian filter kernel applied to the rocks channel

def applyGaussianFilter(rocksImgFiltered):
    """Apply a Gaussian filter to the rocks image to smooth it."""
    rocksGaussianFiltered = np.zeros_like(rocksImgFiltered)  # Initialize the array for OpenCV filtered image

    for zSlice in tqdm(range(rocksImgFiltered.shape[0])):
        rocksGaussianFiltered[zSlice] = cv2.GaussianBlur(rocksImgFiltered[zSlice], (0, 0), sigmaX=GAUSSIAN_FILTER_SIZE, sigmaY=GAUSSIAN_FILTER_SIZE)

    return rocksGaussianFiltered

def applyOtsuThreshold(rocksGaussianFiltered):
    """Apply Otsu's thresholding to the Gaussian filtered rocks image on a slice by slice basis."""
    from skimage.filters import threshold_otsu
    
    rocksOtsuThreshold = np.zeros_like(rocksGaussianFiltered, dtype=bool)
    for zSlice in range(rocksGaussianFiltered.shape[0]):
        # Apply Otsu's thresholding to each zSlice
        otsuThreshold = threshold_otsu(rocksGaussianFiltered[zSlice])
        rocksOtsuThreshold[zSlice] = rocksGaussianFiltered[zSlice] > otsuThreshold

    return rocksOtsuThreshold

<<<<<<< HEAD
def segmentRocksWatershed(rocksOtsuThreshold, zRange):
    """Segment rocks using the watershed algorithm."""

    # Generate markers from the center zSlice as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(rocksOtsuThreshold[zRange[0]:zRange[1]])
    distance_smoothed = ndi.gaussian_filter(distance, sigma=21)
    coords = peak_local_max(distance_smoothed, 
=======
def applyMeanThreshold(rocksGaussianFiltered):
    """Apply mean thresholding to the Gaussian filtered rocks image on a slice by slice basis."""
    rocksMeanThreshold = np.zeros_like(rocksGaussianFiltered, dtype=bool)
    for zSlice in range(rocksGaussianFiltered.shape[0]):
        meanThreshold = threshold_mean(rocksGaussianFiltered[zSlice])
        rocksMeanThreshold[zSlice] = rocksGaussianFiltered[zSlice] > meanThreshold

    return rocksMeanThreshold

def segmentRocksWatershed(rocksOtsuThreshold, zRange):
    """Segment rocks using the watershed algorithm."""

    # Pad the image with zeros (background) on all sides
    rocksPadded = np.pad(rocksOtsuThreshold, pad_width=1, mode='constant', constant_values=0)

    # Generate markers from the center zSlice as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(rocksPadded)
    distance_smoothed = ndi.gaussian_filter(distance, sigma=11)
    coords = peak_local_max(distance_smoothed[zRange[0]:zRange[1]], 
>>>>>>> main
                            min_distance=50, 
                            threshold_rel=0.1, 
                            exclude_border=False, 
                            footprint=np.ones((50, 50, 50)),)
    print(f"Found {len(coords)} local maxima as markers.")

    # Use the markers to fill out the watershed in all zSlices
<<<<<<< HEAD
    mask = np.zeros(rocksOtsuThreshold[zRange[0]:zRange[1]].shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=rocksOtsuThreshold[zRange[0]:zRange[1]])

    return coords, labels

def findCentroids(labels):
    """Find centroids of the segmented rocks."""
    from skimage.measure import regionprops

    centroids = []
    for region in regionprops(labels):
        if region.area > 0:  # Only consider non-empty regions
            centroids.append(region.centroid)
    
    return np.array(centroids)
=======
    mask = np.zeros(rocksPadded[zRange[0]:zRange[1]].shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance[zRange[0]:zRange[1]], markers, mask=rocksPadded[zRange[0]:zRange[1]])

    # Unpad labels and coords
    labels = labels[1:-1, 1:-1, 1:-1]
    coords = coords - 1  # Adjust coords to match unpadded labels

    return coords, labels
>>>>>>> main
    
def __main__():
    #TODO: switch to saving as zarrs instead of tiffs

    # Load the Zarr dataset
    zarrPath = sys.argv[1]
    print(f"Loading Zarr dataset from {zarrPath}...")
    parentDir = os.path.dirname(zarrPath) + "/"
    outputDir = parentDir + "segmentation/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

<<<<<<< HEAD
    # Set time range from command line arguments
    tRange = range(int(sys.argv[2]), int(sys.argv[3]))

    allLabels = []
=======
    # load zarr to set dataset shape
    root = zarr.open(zarrPath, mode='r')
    resArray = root['0']['0'] 
    resArrayShape = resArray.shape # (time, channels, z, y, x)
    del resArray

    # Set time range from command line arguments
    tRange = range(int(sys.argv[2]), int(sys.argv[3]))
    # Define the zRange for segmentation
    zRange = (5, 90) # Adjust this range as needed

    regionProps = []
    uint16Warning = False
>>>>>>> main

    for t in tRange:

        tpOutputDir = outputDir + f"t{t}/"

<<<<<<< HEAD
        # Check if segmentation has already been done
        segmentedRocksPath = tpOutputDir + "segmented_rocks.tif"
        if not os.path.exists(segmentedRocksPath):
            # If not, check if the Otsu thresholded image exists
            rocksOtsuPath = tpOutputDir + "rocks_gaussian_filtered_otsu_threshold.tif"
            if not os.path.exists(rocksOtsuPath):
                # If not, check if the shadows removed image exists
                rocksImgPath = tpOutputDir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
                if not os.path.exists(rocksImgPath):
                    print(f"\nShadows removed image for timepoint {t} does not exist. Skipping...")
                    continue
                else:
                    print(f"Loading shadows removed image for timepoint {t}...")
                    rocksImgFiltered = tiff.imread(rocksImgPath)
                print("Applying Gaussian filter and Otsu's thresholding to the rocks image...")
                # Invert rock channel
                rocksImgFiltered = 65535 - rocksImgFiltered
                # return 0 values to 0
                rocksImgFiltered[rocksImgFiltered == 65535] = 0

                # Apply Gaussian filter to the rocks image
                rocksGaussianFiltered = applyGaussianFilter(rocksImgFiltered)

                # Apply Otsu's thresholding to the Gaussian filtered rocks image
                rocksOtsuThreshold = applyOtsuThreshold(rocksGaussianFiltered)

                #Save the Otsu thresholded image
                tiff.imwrite(rocksOtsuPath, rocksOtsuThreshold.astype("uint16"))
                print(f"Otsu thresholded rocks image saved to {rocksOtsuPath}")

                # Clear memory
                del rocksGaussianFiltered
                del rocksImgFiltered
            else:
                print(f"Otsu thresholded rocks image already exists. Skipping processing for time point {t}.")
                rocksOtsuThreshold = tiff.imread(rocksOtsuPath)
            print("Segmenting rocks using watershed...")

            # Define the zRange for segmentation
            zRange = (4, 90) # Adjust this range as needed
        
            coords, labels = segmentRocksWatershed(rocksOtsuThreshold, zRange)

            # Save the segmented rocks image
            tiff.imwrite(segmentedRocksPath, labels.astype("uint16"))
            print(f"Segmented rocks image saved to {segmentedRocksPath}")
            del rocksOtsuThreshold
        else:
            print(f"Segmented rocks image already exists. Skipping processing for time point {t}.")
            labels = tiff.imread(segmentedRocksPath)

        allLabels.append(labels)

    # Generate tracks from labels
=======
        # Check if centroids have been calculated for this timepoint
        centroidsPath = tpOutputDir + "rock_centroids.csv"
        if not os.path.exists(centroidsPath):
            # If not, check if segmentation has already been done
            segmentedRocksPath = tpOutputDir + "segmented_rocks.tif"
            if not os.path.exists(segmentedRocksPath):
                # If not, check if the Otsu thresholded image exists
                # TODO: switch to mean thresholding
                rocksOtsuPath = tpOutputDir + "rocks_gaussian_filtered_mean_threshold.tif"
                if not os.path.exists(rocksOtsuPath):
                    # If not, check if the shadows removed image exists
                    rocksImgPath = tpOutputDir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
                    if not os.path.exists(rocksImgPath):
                        print(f"\nShadows removed image for timepoint {t} does not exist. Skipping...")
                        continue
                    else:
                        print(f"Loading shadows removed image for timepoint {t}...")
                        rocksImgFiltered = tiff.imread(rocksImgPath)
                    print("Applying Gaussian filter and Otsu's thresholding to the rocks image...")
                    # Invert rock channel
                    rocksImgFiltered = 65535 - rocksImgFiltered
                    # return 0 values to 0
                    rocksImgFiltered[rocksImgFiltered == 65535] = 0

                    # Apply Gaussian filter to the rocks image
                    rocksGaussianFiltered = applyGaussianFilter(rocksImgFiltered)

                    # Apply Otsu's thresholding to the Gaussian filtered rocks image
                    #rocksOtsuThreshold = applyOtsuThreshold(rocksGaussianFiltered)
                    rocksOtsuThreshold = applyMeanThreshold(rocksGaussianFiltered)

                    #Save the Otsu thresholded image
                    tiff.imwrite(rocksOtsuPath, rocksOtsuThreshold.astype("uint16"))
                    print(f"Otsu thresholded rocks image saved to {rocksOtsuPath}")

                    # Clear memory
                    del rocksGaussianFiltered
                    del rocksImgFiltered
                else:
                    print(f"Otsu thresholded rocks image already exists. Skipping processing for time point {t}.")
                    rocksOtsuThreshold = tiff.imread(rocksOtsuPath)
                print("Segmenting rocks using watershed...")
            
                coords, labels = segmentRocksWatershed(rocksOtsuThreshold, zRange)

                # Check if there are more than 256 unique labels
                numLabels = len(np.unique(labels))
                if numLabels > 256:
                    uint16Warning = True
                    print(f"Warning: More than 256 unique labels ({numLabels}) found in the segmented rocks image. Tracked labels will be saved as uint16.")

                # Save the segmented rocks image
                tiff.imwrite(segmentedRocksPath, labels.astype("uint16"))
                csvPath = tpOutputDir + "rock_marker_coords.csv"
                pd.DataFrame(coords, columns=["z", "y", "x"]).to_csv(csvPath, index=False)
                print(f"Rock marker coordinates saved to {csvPath}")
                print(f"Segmented rocks image saved to {segmentedRocksPath}")

                del rocksOtsuThreshold
            else:
                print(f"Segmented rocks image already exists. Skipping processing for time point {t}.")
                labels = tiff.imread(segmentedRocksPath)
                # Check if there are more than 256 unique labels
                numLabels = len(np.unique(labels))
                if numLabels > 256:
                    uint16Warning = True
                    print(f"Warning: More than 256 unique labels ({numLabels}) found in the segmented rocks image. Tracked labels will be saved as uint16.")
            
            # Find centroids of the segmented rocks
            centroids = pd.DataFrame(regionprops_table(labels, properties=['label', 'centroid']))
            centroids['frame'] = t
            regionProps.append(centroids)
            # Save centroids to CSV
            centroids.to_csv(centroidsPath, index=False)
            del labels
        else:
            print(f"Centroids already exist for timepoint {t}. Skipping segmentation.")
            centroids = pd.read_csv(centroidsPath)
            regionProps.append(centroids)

    # Generate tracks from centroids
>>>>>>> main
    tracksPath = outputDir + "rock_tracks.csv"
    graphPath = outputDir + "rock_tracking.graphml"
    if not os.path.exists(tracksPath):
        # Calculate centroids for each label in each frame
        print("Calculating centroids for each label in each frame...")
<<<<<<< HEAD
        regionProps = []
        for frame, label in enumerate(allLabels):
            df = pd.DataFrame(regionprops_table(label, properties=['label', 'centroid']))
            df['frame'] = frame
            regionProps.append(df)
=======

>>>>>>> main
        regionPropsDf = pd.concat(regionProps)

        # Use LapTrack to track the rocks
        print("Tracking rocks across all frames using LapTrack...")
        lt = LapTrack(cutoff=15**2, splitting_cutoff=30**2)
        trackDf, splitDf, mergeDf = lt.predict_dataframe(
            regionPropsDf.copy(),
            coordinate_cols=["centroid-0", "centroid-1", "centroid-2"],
            only_coordinate_cols=False,
        )
        trackDf = trackDf.reset_index()
        graph = convert_split_merge_df_to_napari_graph(splitDf, mergeDf)

        # Save the tracking results
        trackDf.to_csv(tracksPath, index=False)
        if len(graph)!=0:
            print(len(graph))
<<<<<<< HEAD
            nx.write_graphml(graph, graphPath)
=======
            #nx.write_graphml(graph, graphPath)
>>>>>>> main
        print(f"Tracking results saved to {tracksPath} and {graphPath}")
    else:
        print(f"Tracking results already exist at {tracksPath}. Skipping tracking.")
        trackDf = pd.read_csv(tracksPath)
<<<<<<< HEAD
        if os.path.exists(graphPath):
            print(f"Loading tracking graph from {graphPath}...")
            graph = nx.read_graphml(graphPath)

=======
        # if os.path.exists(graphPath):
        #     print(f"Loading tracking graph from {graphPath}...")
        #     graph = nx.read_graphml(graphPath)

    # TODO: Go frame by frame and write trackedLabels to a zarr to save memory
>>>>>>> main
    # Link labels across frames
    trackedLabelsPath = outputDir + "tracked_labels.tiff"
    if not os.path.exists(trackedLabelsPath):
        print("Using tracks to link labels across frames...")
<<<<<<< HEAD
        trackedLabels = np.zeros_like(allLabels)

        for i, row in trackDf.iterrows():
            frame = int(row["frame"])
            inds = allLabels[frame] == row["label"]
            trackedLabels[frame][inds] = int(row["tree_id"]) + 1
        tiff.imwrite(trackedLabelsPath, trackedLabels.astype("uint16"))
=======
        # Generate an array to hold the tracked labels, must be same shape as the original dataset (time, z, y, x) to be imported into Imaris
        # if uint16Warning:
        trackedLabels = np.zeros((resArrayShape[0], resArrayShape[2], resArrayShape[3], resArrayShape[4]), dtype="uint16") # (time, z, y, x)
        # else:
        #     trackedLabels = np.zeros((resArrayShape[0], resArrayShape[2], resArrayShape[3], resArrayShape[4]), dtype="uint8")
        for t in tRange:
            print(f"Linking labels for timepoint {t}...")
            labelsPath = outputDir + f"t{t}/segmented_rocks.tif"
            labels = tiff.imread(labelsPath)
            for i, row in trackDf[trackDf["frame"] == t].iterrows():
                inds = labels == row["label"]
                zRange = (4, 4+inds.shape[0]) # TODO: Remove in the future when zRange is consistent for all frames
                trackedLabels[t][zRange[0]:zRange[1]][inds] = int(row["tree_id"]) + 1
            del labels

        # Save the tracked labels
        tiff.imwrite(trackedLabelsPath, trackedLabels, imagej=True, metadata={'axes': 'TZYX'})
>>>>>>> main
    else:
        print(f"Tracked labels already exist at {trackedLabelsPath}. Skipping linking.")
        trackedLabels = tiff.imread(trackedLabelsPath)

    # Calculate track vectors for each frame pair
    motion_vectors = []
    for track_id in trackDf["track_id"].unique():
        track = trackDf[trackDf["track_id"] == track_id]
        if len(track) > 1:
            # Calculate motion vectors for each frame pair
            track_motion_vectors = []
            for i in range(len(track) - 1):
                start = track.iloc[i]
                end = track.iloc[i + 1]
                frame_vector = pd.DataFrame({
                    "motion_z": [end["centroid-0"] - start["centroid-0"]],
                    "motion_y": [end["centroid-1"] - start["centroid-1"]],
                    "motion_x": [end["centroid-2"] - start["centroid-2"]],
                    "track_id": [track_id],
                    "frame": [int(start["frame"])+1]
                })
                track_motion_vectors.append(frame_vector)
            track_motion_vectors = pd.concat(track_motion_vectors, ignore_index=True)
            motion_vectors.append(track_motion_vectors)

    motion_vectors = pd.concat(motion_vectors, ignore_index=True)

    # Calculate magnitude of motion vectors for each frame pair
    motion_vectors["magnitude"] = np.sqrt(motion_vectors["motion_x"]**2 + 
                                       motion_vectors["motion_y"]**2 + 
                                       motion_vectors["motion_z"]**2)
    
    # Calculate and sort tracks by total motion from first to last frame
    motion_vectors["total_motion"] = motion_vectors.groupby("track_id")["magnitude"].transform("sum")
    motion_vectors = motion_vectors.sort_values(by="total_motion", ascending=False).reset_index(drop=True)

    # Save motion vectors
    motionVectorsPath = outputDir + "rock_motion_vectors.csv"
    motion_vectors.to_csv(motionVectorsPath, index=False)
    print(f"Motion vectors saved to {motionVectorsPath}")


if __name__ == "__main__":
    __main__()
<<<<<<< HEAD

=======
>>>>>>> main
