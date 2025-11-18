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
from skimage.filters import threshold_mean
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
    """Apply Otsu's thresholding to the Gaussian filtered rocks image on a slice by slice basis."""
    from skimage.filters import threshold_otsu
    
    rocks_otsu_threshold = np.zeros_like(rocks_gaussian_filtered, dtype=bool)
    for z_slice in range(rocks_gaussian_filtered.shape[0]):
        # Apply Otsu's thresholding to each z_slice
        otsu_threshold = threshold_otsu(rocks_gaussian_filtered[z_slice])
        rocks_otsu_threshold[z_slice] = rocks_gaussian_filtered[z_slice] > otsu_threshold

    return rocks_otsu_threshold

def apply_mean_threshold(rocks_gaussian_filtered):
    """Apply mean thresholding to the Gaussian filtered rocks image on a slice by slice basis."""
    rocks_mean_threshold = np.zeros_like(rocks_gaussian_filtered, dtype=bool)
    for z_slice in range(rocks_gaussian_filtered.shape[0]):
        mean_threshold = threshold_mean(rocks_gaussian_filtered[z_slice])
        rocks_mean_threshold[z_slice] = rocks_gaussian_filtered[z_slice] > mean_threshold

    return rocks_mean_threshold

def segment_rocks_watershed(rocks_otsu_threshold, z_range):
    """Segment rocks using the watershed algorithm."""

    # Pad the image with zeros (background) on all sides
    rocks_padded = np.pad(rocks_otsu_threshold, pad_width=1, mode='constant', constant_values=0)

    # Generate markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(rocks_padded)
    distance_smoothed = ndi.gaussian_filter(distance, sigma=11)
    coords = peak_local_max(distance_smoothed[z_range[0]:z_range[1]], 
                            min_distance=50, 
                            threshold_rel=0.1, 
                            exclude_border=False, 
                            footprint=np.ones((50, 50, 50)),)
    print(f"Found {len(coords)} local maxima as markers.")

    # Use the markers to fill out the watershed in all z_slices
    mask = np.zeros(rocks_padded[z_range[0]:z_range[1]].shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance[z_range[0]:z_range[1]], markers, mask=rocks_padded[z_range[0]:z_range[1]])

    # Unpad labels and coords
    labels = labels[1:-1, 1:-1, 1:-1]
    coords = coords - 1  # Adjust coords to match unpadded labels

    return coords, labels
    
def __main__():
    #TODO: switch to saving as zarrs instead of tiffs

    # Load the Zarr dataset
    zarr_path = sys.argv[1]
    print(f"Loading Zarr dataset from {zarr_path}...")
    parent_dir = os.path.dirname(zarr_path) + "/"
    output_dir = parent_dir + "segmentation/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load zarr to set dataset shape
    root = zarr.open(zarr_path, mode='r')
    res_array = root['0']['0'] 
    res_array_shape = res_array.shape # (time, channels, z, y, x)
    del res_array

    # Set time range from command line arguments
    t_range = range(int(sys.argv[2]), int(sys.argv[3]))
    # Define the z_range for segmentation
    z_range = (5, 90) # Adjust this range as needed

    region_props = []
    uint16_warning = False

    for t in t_range:

        tp_output_dir = output_dir + f"t{t}/"

        # Check if centroids have been calculated for this timepoint
        centroids_path = tp_output_dir + "rock_centroids.csv"
        if not os.path.exists(centroids_path):
            # If not, check if segmentation has already been done
            segmented_rocks_path = tp_output_dir + "segmented_rocks.tif"
            if not os.path.exists(segmented_rocks_path):
                # If not, check if the Otsu thresholded image exists
                # TODO: switch to mean thresholding
                rocks_otsu_path = tp_output_dir + "rocks_gaussian_filtered_mean_threshold.tif"
                if not os.path.exists(rocks_otsu_path):
                    # If not, check if the shadows removed image exists
                    rocks_img_path = tp_output_dir + "cell_shadows_removed_" + str(THRESHOLD) + "thresh_" + str(SHADOW_MED_FILT) + "medfilt.tif"
                    if not os.path.exists(rocks_img_path):
                        print(f"\nShadows removed image for timepoint {t} does not exist. Skipping...")
                        continue
                    else:
                        print(f"Loading shadows removed image for timepoint {t}...")
                        rocks_img_filtered = tiff.imread(rocks_img_path)
                    print("Applying Gaussian filter and Otsu's thresholding to the rocks image...")
                    # Invert rock channel
                    rocks_img_filtered = 65535 - rocks_img_filtered
                    # return 0 values to 0
                    rocks_img_filtered[rocks_img_filtered == 65535] = 0

                    # Apply Gaussian filter to the rocks image
                    rocks_gaussian_filtered = apply_gaussian_filter(rocks_img_filtered)

                    # Apply Otsu's thresholding to the Gaussian filtered rocks image
                    #rocks_otsu_threshold = apply_otsu_threshold(rocks_gaussian_filtered)
                    rocks_otsu_threshold = apply_mean_threshold(rocks_gaussian_filtered)

                    #Save the Otsu thresholded image
                    tiff.imwrite(rocks_otsu_path, rocks_otsu_threshold.astype("uint16"))
                    print(f"Otsu thresholded rocks image saved to {rocks_otsu_path}")

                    # Clear memory
                    del rocks_gaussian_filtered
                    del rocks_img_filtered
                else:
                    print(f"Otsu thresholded rocks image already exists. Skipping processing for time point {t}.")
                    rocks_otsu_threshold = tiff.imread(rocks_otsu_path)
                print("Segmenting rocks using watershed...")
            
                coords, labels = segment_rocks_watershed(rocks_otsu_threshold, z_range)

                # Check if there are more than 256 unique labels
                num_labels = len(np.unique(labels))
                if num_labels > 256:
                    uint16_warning = True
                    print(f"Warning: More than 256 unique labels ({num_labels}) found in the segmented rocks image. Tracked labels will be saved as uint16.")

                # Save the segmented rocks image
                tiff.imwrite(segmented_rocks_path, labels.astype("uint16"))
                csv_path = tp_output_dir + "rock_marker_coords.csv"
                pd.DataFrame(coords, columns=["z", "y", "x"]).to_csv(csv_path, index=False)
                print(f"Rock marker coordinates saved to {csv_path}")
                print(f"Segmented rocks image saved to {segmented_rocks_path}")

                del rocks_otsu_threshold
            else:
                print(f"Segmented rocks image already exists. Skipping processing for time point {t}.")
                labels = tiff.imread(segmented_rocks_path)
                # Check if there are more than 256 unique labels
                num_labels = len(np.unique(labels))
                if num_labels > 256:
                    uint16_warning = True
                    print(f"Warning: More than 256 unique labels ({num_labels}) found in the segmented rocks image. Tracked labels will be saved as uint16.")
            
            # Find centroids of the segmented rocks
            centroids = pd.DataFrame(regionprops_table(labels, properties=['label', 'centroid']))
            centroids['frame'] = t
            region_props.append(centroids)
            # Save centroids to CSV
            centroids.to_csv(centroids_path, index=False)
            del labels
        else:
            print(f"Centroids already exist for timepoint {t}. Skipping segmentation.")
            centroids = pd.read_csv(centroids_path)
            region_props.append(centroids)

    # Generate tracks from centroids
    tracks_path = output_dir + "rock_tracks.csv"
    graph_path = output_dir + "rock_tracking.graphml"
    if not os.path.exists(tracks_path):
        # Calculate centroids for each label in each frame
        print("Calculating centroids for each label in each frame...")

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
            #nx.write_graphml(graph, graph_path)
        print(f"Tracking results saved to {tracks_path} and {graph_path}")
    else:
        print(f"Tracking results already exist at {tracks_path}. Skipping tracking.")
        track_df = pd.read_csv(tracks_path)
        # if os.path.exists(graph_path):
        #     print(f"Loading tracking graph from {graph_path}...")
        #     graph = nx.read_graphml(graph_path)

    # TODO: Go frame by frame and write tracked_labels to a zarr to save memory
    # Link labels across frames
    tracked_labels_path = output_dir + "tracked_labels.tiff"
    if not os.path.exists(tracked_labels_path):
        print("Using tracks to link labels across frames...")
        # Generate an array to hold the tracked labels, must be same shape as the original dataset (time, z, y, x) to be imported into Imaris
        # if uint16_warning:
        tracked_labels = np.zeros((res_array_shape[0], res_array_shape[2], res_array_shape[3], res_array_shape[4]), dtype="uint16") # (time, z, y, x)
        # else:
        #     tracked_labels = np.zeros((res_array_shape[0], res_array_shape[2], res_array_shape[3], res_array_shape[4]), dtype="uint8")
        for t in t_range:
            print(f"Linking labels for timepoint {t}...")
            labels_path = output_dir + f"t{t}/segmented_rocks.tif"
            labels = tiff.imread(labels_path)
            for i, row in track_df[track_df["frame"] == t].iterrows():
                inds = labels == row["label"]
                z_range = (4, 4+inds.shape[0]) # TODO: Remove in the future when z_range is consistent for all frames
                tracked_labels[t][z_range[0]:z_range[1]][inds] = int(row["tree_id"]) + 1
            del labels

        # Save the tracked labels
        tiff.imwrite(tracked_labels_path, tracked_labels, imagej=True, metadata={'axes': 'TZYX'})
    else:
        print(f"Tracked labels already exist at {tracked_labels_path}. Skipping linking.")
        tracked_labels = tiff.imread(tracked_labels_path)

    # Calculate track vectors for each frame pair
    motion_vectors = []
    for track_id in track_df["track_id"].unique():
        track = track_df[track_df["track_id"] == track_id]
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
    motion_vectors_path = output_dir + "rock_motion_vectors.csv"
    motion_vectors.to_csv(motion_vectors_path, index=False)
    print(f"Motion vectors saved to {motion_vectors_path}")


if __name__ == "__main__":
    __main__()
