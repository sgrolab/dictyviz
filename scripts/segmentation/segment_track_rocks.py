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
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from laptrack import LapTrack
from laptrack.data_conversion import convert_split_merge_df_to_napari_graph

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal
GAUSSIAN_FILTER_SIZE = 2.2  # standard deviation of the Gaussian filter kernel applied to the rocks channel

def apply_gaussian_filter(rocks_img_filtered):
    """Apply a Gaussian filter to the rocks image to smooth it."""
    rocks_gaussian_filtered = np.zeros_like(rocks_img_filtered)  # Initialize the array for OpenCV filtered image

    for z_slice in tqdm(range(rocks_img_filtered.shape[0])):
        rocks_gaussian_filtered[z_slice] = cv2.GaussianBlur(rocks_img_filtered[z_slice], (0, 0), sigmaX=GAUSSIAN_FILTER_SIZE, sigmaY=GAUSSIAN_FILTER_SIZE)

    return rocks_gaussian_filtered

def apply_otsu_threshold(rocks_gaussian_filtered):
    """Apply Otsu's thresholding to the Gaussian filtered rocks image."""
    from skimage.filters import threshold_otsu

    otsu_threshold = threshold_otsu(rocks_gaussian_filtered)
    rocks_otsu_threshold = rocks_gaussian_filtered > otsu_threshold

    return rocks_otsu_threshold

def segment_rocks_watershed(rocks_otsu_threshold, z_range):
    """Segment rocks using the watershed algorithm."""

    middle_slice = (z_range[0] + z_range[1]) // 2

    # Generate markers from the center z_slice as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(rocks_otsu_threshold[z_range[0]:z_range[1]])
    distance_smoothed = ndi.gaussian_filter(distance, sigma=21)
    coords = peak_local_max(distance_smoothed[middle_slice], 
                            min_distance=50, 
                            threshold_rel=0.1, 
                            exclude_border=False, 
                            footprint=np.ones((50, 50)))
    print(f"Found {len(coords)} local maxima as markers.")

    # Use the markers to fill out the watershed in all z_slices
    mask = np.zeros(rocks_otsu_threshold[z_range[0]:z_range[1]].shape, dtype=bool)
    mask[middle_slice][tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=rocks_otsu_threshold[z_range[0]:z_range[1]])

    return coords, labels

def find_centroids(labels):
    """Find centroids of the segmented rocks."""
    from skimage.measure import regionprops

    centroids = []
    for region in regionprops(labels):
        if region.area > 0:  # Only consider non-empty regions
            centroids.append(region.centroid)
    
    return np.array(centroids)

def calculate_rock_tracks(centroids):
    """Use linear sum assignment to calculate rock tracks."""
    from scipy.optimize import linear_sum_assignment

    for t in range(centroids.shape[0] - 1):
        # Calculate the cost matrix for linear sum assignment
        cost_matrix = np.linalg.norm(centroids[t][:, np.newaxis] - centroids[t + 1], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        tracks = np.column_stack((row_ind, col_ind))
        print(f"Frame {t} to {t + 1} tracks: {tracks}")
    

def __main__():
    #TODO: switch to saving as zarrs instead of tiffs

    # Load the Zarr dataset
    zarr_path = sys.argv[1]
    print(f"Loading Zarr dataset from {zarr_path}...")
    parent_dir = os.path.dirname(zarr_path) + "/"
    output_dir = parent_dir + "segmentation/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set time range from command line arguments
    t_range = range(int(sys.argv[2]), int(sys.argv[3]))

    all_labels = []

    for t in t_range:

        # TODO: Reorder logic so that it checks for the final output first and works backwards
        # This way it can skip loading earlier steps if processing is already complete

        tp_output_dir = output_dir + f"t{t}/"
        rocks_img_path = tp_output_dir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
        if not os.path.exists(rocks_img_path):
            print(f"\nShadows removed image for timepoint {t} does not exist. Skipping...")
            continue
        else:
            print(f"Loading shadows removed image for timepoint {t}...")
            rocks_img_filtered = tiff.imread(rocks_img_path)
            #cells_img_masked = tiff.imread(cells_img_path)

        rocks_otsu_path = tp_output_dir + "rocks_gaussian_filtered_otsu_threshold.tif"
        if not os.path.exists(rocks_otsu_path):
            print("Applying Gaussian filter and Otsu's thresholding to the rocks image...")
            # Invert rock channel
            rocks_img_filtered = 65535 - rocks_img_filtered
            # return 0 values to 0
            rocks_img_filtered[rocks_img_filtered == 65535] = 0

            # Apply Gaussian filter to the rocks image
            rocks_gaussian_filtered = apply_gaussian_filter(rocks_img_filtered)

            # Apply Otsu's thresholding to the Gaussian filtered rocks image
            rocks_otsu_threshold = apply_otsu_threshold(rocks_gaussian_filtered)

            #Save the Otsu thresholded image
            tiff.imwrite(rocks_otsu_path, rocks_otsu_threshold.astype("uint16"))
            print(f"Otsu thresholded rocks image saved to {rocks_otsu_path}")

            # Clear memory
            del rocks_gaussian_filtered
            del rocks_img_filtered
        else:
            print(f"Otsu thresholded rocks image already exists. Skipping processing for time point {t}.")
            rocks_otsu_threshold = tiff.imread(rocks_otsu_path)

        segmented_rocks_path = tp_output_dir + "segmented_rocks.tif"
        if not os.path.exists(segmented_rocks_path):
            print("Segmenting rocks using watershed...")

            # Define the z_range for segmentation
            z_range = (4, 100) # Adjust this range as needed
        
            coords, labels = segment_rocks_watershed(rocks_otsu_threshold, z_range)

            # Save the segmented rocks image
            tiff.imwrite(segmented_rocks_path, labels.astype("uint16"))
            print(f"Segmented rocks image saved to {segmented_rocks_path}")
            del rocks_otsu_threshold
        else:
            print(f"Segmented rocks image already exists. Skipping processing for time point {t}.")
            labels = tiff.imread(segmented_rocks_path)
            del rocks_otsu_threshold

        all_labels.append(labels)

    tracks_path = output_dir + "rock_tracks.csv"
    graph_path = output_dir + "rock_tracking.graphml"
    if not os.path.exists(tracks_path):
        # Calculate centroids for each label in each frame
        print("Calculating centroids for each label in each frame...")
        region_props = []
        for frame, label in enumerate(all_labels):
            df = pd.DataFrame(regionprops_table(label, properties=['label', 'centroid']))
            df['frame'] = frame
            region_props.append(df)
        region_props_df = pd.concat(region_props)

        # Use LapTrack to track the rocks
        print("Tracking rocks across all frames using LapTrack...")
        lt = LapTrack(cutoff=15**2, splitting_cutoff=30**2)
        track_df, split_df, merge_df = lt.predict_dataframe(
            region_props_df.copy(),
            coordinate_cols=["centroid-0", "centroid-1", "centroid-2"],
            only_coordinate_cols=False,
        )
        track_df = track_df.reset_index()
        graph = convert_split_merge_df_to_napari_graph(split_df, merge_df)

        # Save the tracking results
        track_df.to_csv(tracks_path, index=False)
        if len(graph)!=0:
            print(len(graph))
            nx.write_graphml(graph, graph_path)
        print(f"Tracking results saved to {tracks_path} and {graph_path}")
    else:
        print(f"Tracking results already exist at {tracks_path}. Skipping tracking.")
        track_df = pd.read_csv(tracks_path)
        if os.path.exists(graph_path):
            print(f"Loading tracking graph from {graph_path}...")
            graph = nx.read_graphml(graph_path)

    # Link labels across frames
    tracked_labels_path = output_dir + "tracked_labels.tiff"
    if not os.path.exists(tracked_labels_path):
        print("Using tracks to link labels across frames...")
        tracked_labels = np.zeros_like(all_labels)

        for i, row in track_df.iterrows():
            frame = int(row["frame"])
            inds = all_labels[frame] == row["label"]
            tracked_labels[frame][inds] = int(row["tree_id"]) + 1
        tiff.imwrite(tracked_labels_path, tracked_labels.astype("uint16"))
    else:
        print(f"Tracked labels already exist at {tracked_labels_path}. Skipping linking.")
        tracked_labels = tiff.imread(tracked_labels_path)

    # Calculate track vectors for each frame pair
    track_vectors = []


if __name__ == "__main__":
    __main__()

