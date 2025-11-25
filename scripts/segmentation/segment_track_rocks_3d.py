import os
import sys
import zarr
import cv2
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import dask.array as da
import networkx as nx

from scipy import ndimage as ndi
from skimage.filters import threshold_mean
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops_table
from laptrack import LapTrack
from laptrack.data_conversion import convert_split_merge_df_to_napari_graph

THRESHOLD = 98 # Intensity threshold (percentile) in the cell channel for shadow removal
SHADOW_MED_FILT = 3 # Size of median filter for shadow removal
GAUSSIAN_FILTER_SIZE = 2.2  # standard deviation of the Gaussian filter kernel applied to the rocks channel

def apply_gaussian_filter(rocks_img_filtered):
    """Apply a Gaussian filter to the rocks image to smooth it."""
    rocks_gaussian_filtered = np.zeros_like(rocks_img_filtered)  # Initialize the array for OpenCV filtered image

    for z_slice in range(rocks_img_filtered.shape[0]):
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
    coords = peak_local_max(distance_smoothed, 
                            min_distance=50, 
                            threshold_rel=0.1, 
                            exclude_border=False, 
                            footprint=np.ones((50, 50, 50)),)
    print(f"Found {len(coords)} local maxima as markers.")

    # Use the markers to fill out the watershed in all z_slices
    mask = np.zeros(rocks_padded.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=rocks_padded)

    # Unpad labels and coords
    labels = labels[1:-1, 1:-1, 1:-1]
    coords = coords - 1  # Adjust coords to match unpadded labels

    return coords, labels


def create_boundary_rag(labels):
    """
    Create a region adjacency graph where edge weights are the fraction of 
    boundary surface shared relative to the smaller region's surface area.
    
    Parameters
    ----------
    labels : ndarray
        Labeled array where each region has a unique integer label.
        
    Returns
    -------
    graph : networkx.Graph
        Region adjacency graph where edge weights represent boundary fraction.
    node_surface_areas : dict
        Dictionary mapping region labels to their surface areas.
    """
    
    # Find boundaries between regions
    boundaries = find_boundaries(labels, mode='inner')
    
    # Create graph
    g = nx.Graph()
    
    # Add all unique labels as nodes
    unique_labels = np.unique(labels[labels > 0])  # Exclude background (0)
    g.add_nodes_from(unique_labels)
    
    # Dictionary to count shared boundaries between region pairs
    edge_counts = {}
    
    # Dictionary to count surface area for each region
    surface_counts = {label: 0 for label in unique_labels}
    
    print("Calculating boundary adjacencies and surface areas...")
    
    # Find all boundary pixels and their neighboring regions
    boundary_coords = np.argwhere(boundaries)
    
    for coord in boundary_coords:
        # Get the label at this boundary pixel
        current_label = labels[tuple(coord)]
        
        if current_label == 0:  # Skip background
            continue
        
        z, y, x = coord
        
        # Check each of the 6 neighboring faces (for 3D)
        neighbors_at_faces = []
        face_directions = [
            (z-1, y, x), (z+1, y, x),  # z neighbors
            (z, y-1, x), (z, y+1, x),  # y neighbors
            (z, y, x-1), (z, y, x+1)   # x neighbors
        ]
        
        for n_z, n_y, n_x in face_directions:
            # Check if neighbor is in bounds
            if (0 <= n_z < labels.shape[0] and 
                0 <= n_y < labels.shape[1] and 
                0 <= n_x < labels.shape[2]):
                neighbor_label = labels[n_z, n_y, n_x]
                
                if neighbor_label == 0:
                    # Face touching background - counts toward surface area
                    surface_counts[current_label] += 1
                elif neighbor_label != current_label:
                    # Face touching different region - counts toward both surface area and shared boundary
                    surface_counts[current_label] += 1
                    neighbors_at_faces.append(neighbor_label)
            else:
                # Face at image boundary - counts toward surface area
                surface_counts[current_label] += 1
        
        # Count shared boundaries with neighbors
        unique_neighbors = set(neighbors_at_faces)
        for neighbor in unique_neighbors:
            region1, region2 = sorted([current_label, neighbor])
            edge_key = (region1, region2)
            # Count the number of faces shared (each shared face counted from both sides)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + neighbors_at_faces.count(neighbor)
    
    print(f"Calculated surface areas for {len(surface_counts)} regions.")
    print(f"Found {len(edge_counts)} region adjacencies.")
    
    # Add edges to graph with boundary fraction as weights
    for (region1, region2), boundary_face_count in edge_counts.items():
        surface1 = surface_counts[region1]
        surface2 = surface_counts[region2]

        # Add surface areas as node properties
        g.nodes[region1]['surface_area'] = surface1
        g.nodes[region2]['surface_area'] = surface2
        
        if surface1 == 0 or surface2 == 0:
            continue
        
        # Calculate fraction relative to smaller region's surface
        # Divide by 2 because each shared face is counted twice (once from each side)
        shared_faces = boundary_face_count / 2
        smaller_surface = min(surface1, surface2)
        boundary_fraction = shared_faces / smaller_surface
        print(f"Regions {region1} and {region2}: boundary fraction = {boundary_fraction:.3f} ({int(shared_faces)} shared faces)")
        
        # Store both the fraction and the raw counts as edge attributes
        g.add_edge(region1, region2, 
                   weight=boundary_fraction,
                   shared_faces=shared_faces,
                   boundary_count=boundary_face_count)
    
    return g


def merge_regions_by_boundary_fraction(labels, rag, boundary_fraction_threshold=0.15):
    """
    Merge regions based on the fraction of boundary surface shared relative to 
    the smaller region's total surface area.
    
    Parameters
    ----------
    labels : ndarray
        Labeled segmentation array.
    rag : networkx.Graph
        Region adjacency graph with edge weights as boundary fractions.
    surface_areas : dict
        Dictionary mapping region labels to their surface areas.
    boundary_fraction_threshold : float
        Minimum fraction of the smaller region's surface area that must be 
        shared boundary for the regions to be merged.
        
    Returns
    -------
    merged_labels : ndarray
        Labels after merging regions based on boundary surface fraction.
    """
    print(f"Starting with {rag.number_of_nodes()} regions.")
    print(f"Boundary fraction threshold: {boundary_fraction_threshold}")
    
    # Create a mapping for merged labels
    label_map = {label: label for label in rag.nodes()}
    
    # Collect all edges with their boundary fractions
    edge_data = []
    for region1, region2, data in rag.edges(data=True):
        boundary_fraction = data['weight']
        shared_faces = data['shared_faces']
        surface1 = rag.nodes[region1]['surface_area']
        surface2 = rag.nodes[region2]['surface_area']
        
        edge_data.append((boundary_fraction, shared_faces, region1, region2, surface1, surface2))
    
    # Sort by boundary fraction (highest first)
    edge_data.sort(reverse=True)
    
    print(f"Processing {len(edge_data)} edges for potential merging...")
    merge_count = 0
    merged_labels = labels.copy()

    for boundary_fraction, shared_faces, region1, region2, surface1, surface2 in edge_data:
        # Skip if below threshold
        if boundary_fraction < boundary_fraction_threshold:
            break  # Since sorted, all remaining are below threshold
        
        # Get current labels (in case they've been remapped)
        current_label1 = label_map[region1]
        current_label2 = label_map[region2]
        
        # Skip if already merged into same region
        if current_label1 == current_label2:
            continue
        
        # Simply merge region1 into region2
        # Remap all instances of region1's current label to region2's current label
        merged_labels[merged_labels == current_label1] = current_label2
        
        merge_count += 1
        if merge_count <= 10:  # Only print first 10 merges
            print(f"  Merging region {region1} (surface={surface1}) "
                  f"into region {region2} (surface={surface2}): "
                  f"boundary fraction={boundary_fraction:.3f} ({int(shared_faces)} shared faces)")
    
    print(f"Merged {merge_count} region pairs.")
    
    # Apply label mapping
    # merged_labels = labels.copy()
    # for old_label, new_label in label_map.items():
    #     print(f"Mapping label {old_label} to {new_label}")
    #     merged_labels[labels == old_label] = new_label
    
    # Relabel to make sequential
    # merged_labels = ndi.label(merged_labels > 0)[0]
    
    final_regions = len(np.unique(merged_labels)) - 1  # Subtract background
    print(f"Final number of regions: {final_regions}")
    
    return merged_labels

    
def __main__(args):

    # Load the Zarr dataset
    zarr_path = args.zarr_path
    print(f"Loading Zarr dataset from {zarr_path}...")

    thresh_method = args.thresh_method  # 'otsu' or 'mean'
    if args.z_range is not None:
        z_range = (args.z_range[0], args.z_range[1])  # Optional z range argument
    if thresh_method not in ['otsu', 'mean']:
        raise ValueError("thresh_method must be either 'otsu' or 'mean'")
    boundary_fraction_threshold = args.boundary_fraction_threshold

    exp_name = f"{z_range[0]}_{z_range[1]}_{thresh_method}_thresh_{boundary_fraction_threshold}_boundary_frac"
    parent_dir = os.path.dirname(zarr_path) + "/"
    seg_dir = parent_dir + "segmentation/"
    output_dir = seg_dir + exp_name + "/"
    print(f"Writing output to experiment directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_root = zarr.open(seg_dir, mode='a')
    rocks_img = seg_root["shadows_removed"]
    T, _, Y, X = rocks_img.shape # (time, z, y, x)
    Z = z_range[1] - z_range[0]
    print(f"Dataset shape: T={T}, Z={Z}, Y={Y}, X={X}")

    # Create output zarrs
    output_root = zarr.open(output_dir, mode='a')
    if thresh_method == 'otsu':
        if os.path.exists(output_dir + "rocks_otsu_thresh"):
            rocks_thresh = output_root["rocks_otsu_thresh"]
        else:
            rocks_thresh = output_root.create_dataset(
                "rocks_otsu_thresh", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint8', overwrite=True
            )
    elif thresh_method == 'mean':
        if os.path.exists(output_dir + "rocks_mean_thresh"):
            rocks_thresh = output_root["rocks_mean_thresh"]
        else:
            rocks_thresh = output_root.create_dataset(
                "rocks_mean_thresh", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint8', overwrite=True
            )
    if os.path.exists(output_dir + "segmented_rocks"):
        segmented_rocks = output_root["segmented_rocks"]
    else:
        segmented_rocks = output_root.create_dataset(
            "segmented_rocks", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint16', overwrite=True
        )
    if os.path.exists(output_dir + "segmented_rocks_merge"):
        segmented_rocks_merge = output_root["segmented_rocks_merge"]
    else:
        segmented_rocks_merge = output_root.create_dataset(
            "segmented_rocks_merge", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint16', overwrite=True
        )
    if os.path.exists(output_dir + "centroids"):
        centroids = output_root["centroids"]
    else:
        centroids = output_root.create_dataset(
            "centroids", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint16', overwrite=True
        )
    if os.path.exists(output_dir + "tracked_labels"):
        tracked_labels = output_root["tracked_labels"]
    else:
        tracked_labels = output_root.create_dataset(
            "tracked_labels", shape=(T, Z, Y, X), chunks=(1, 1, Y, X), dtype='uint16', overwrite=True
        )

    for t in range(T):
        # Check if merging has already been done
        if np.all(segmented_rocks_merge[t] == 0):
            print(f"\nProcessing timepoint {t}...")
            # Check if segmentation has already been done
            if np.all(segmented_rocks[t] == 0):
                # If not, check if the Otsu thresholded image exists
                if np.all(rocks_thresh[t] == 0):
                    # If not, check if the shadows removed image exists
                    rocks_img_tp = rocks_img[t]
                    if rocks_img_tp is None:
                        print(f"\nShadows removed image for timepoint {t} does not exist. Skipping...")
                        continue
                    else:
                        print(f"Loading shadows removed image for timepoint {t}...")
                        print(f"rocks_img_tp shape: {rocks_img_tp.shape}")
                        rocks_img_filtered = rocks_img[t][z_range[0]:z_range[1]]
                    print("Applying Gaussian filter and thresholding rocks image...")
                    # Invert rock channel
                    rocks_img_filtered = 65535 - rocks_img_filtered
                    # return 0 values to 0
                    rocks_img_filtered[rocks_img_filtered == 65535] = 0

                    # Apply Gaussian filter to the rocks image
                    rocks_gaussian_filtered = apply_gaussian_filter(rocks_img_filtered)

                    # Apply thresholding method
                    if thresh_method == 'otsu':
                        rocks_threshold = apply_otsu_threshold(rocks_gaussian_filtered)
                        rocks_thresh[t] = rocks_threshold.astype("uint8")
                        print(f"Otsu thresholded rocks image saved to {output_dir}/rocks_otsu_thresh")
                    elif thresh_method == 'mean':
                        rocks_threshold = apply_mean_threshold(rocks_gaussian_filtered)
                        rocks_thresh[t] = rocks_threshold.astype("uint8")
                        print(f"Mean thresholded rocks image saved to {output_dir}/rocks_mean_thresh")

                    # Clear memory
                    del rocks_gaussian_filtered
                    del rocks_img_filtered
                else:
                    print(f"Otsu thresholded rocks image already exists. Skipping processing for time point {t}.")
                    rocks_threshold = rocks_thresh[t]
                print("Segmenting rocks using watershed...")

                coords, labels = segment_rocks_watershed(rocks_threshold, z_range)

                # Save the segmented rocks image before merging
                segmented_rocks[t] = labels.astype("uint16")
                print(f"Segmented rocks image saved to {output_dir}/segmented_rocks")

                del rocks_threshold

            else:
                print(f"Segmented rocks image already exists. Skipping processing for time point {t}.")
                labels = segmented_rocks[t]
            
            # Use region adjacency graph with weights as boundary surface fraction to merge regions
            print("Creating region adjacency graph...")
            rag = create_boundary_rag(labels)
            
            print("Merging regions based on boundary surface fraction...")
            merged_labels = merge_regions_by_boundary_fraction(labels, rag, boundary_fraction_threshold=0.15)

            # Save the segmented rocks image after merging
            segmented_rocks_merge[t] = merged_labels.astype("uint16")
            print(f"Segmented rocks image saved to {output_dir}/segmented_rocks_merge")

            del labels
            del merged_labels


    # Calculate centroids for each timepoint
    if not os.path.exists(output_dir + "centroids.csv"):
        print("Calculating centroids for each timepoint...")
        region_props = []
        for t in range(T):
            labels = segmented_rocks_merge[t]
            centroids = pd.DataFrame(regionprops_table(labels, properties=['label', 'centroid']))
            centroids['frame'] = t
            region_props.append(centroids)
        region_props_df = pd.concat(region_props)
        region_props_df.to_csv(output_dir + "centroids.csv", index=False)
    else:
        print("Loading centroids from existing file...")
        region_props_df = pd.read_csv(output_dir + "centroids.csv")

    # Generate tracks from centroids
    tracks_path = output_dir + "rock_tracks.csv"
    graph_path = output_dir + "rock_tracking.graphml"
    if not os.path.exists(tracks_path):

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

    # Link labels across frames
    if np.all(tracked_labels) == 0:
        print("Using tracks to link labels across frames...")
        for t in range(T):
            print(f"Linking labels for timepoint {t}...")
            labels = da.from_zarr(output_root["segmented_rocks"])[t].compute()
            # Load the current timepoint into memory
            tracked_t = tracked_labels[t][:]  # Get as numpy array
            for i, row in track_df[track_df["frame"] == t].iterrows():
                inds = np.where(labels == row["label"])
                tracked_t[inds] = int(row["tree_id"]) + 1  # Modify numpy array
            # Write back to zarr
            tracked_labels[t] = tracked_t  # Write the entire timepoint back
            del labels

        # Save the tracked labels
        tracked_labels[:] = tracked_labels
    else:
        print(f"Tracked labels already exist at {output_dir}/tracked_labels. Skipping linking.")
        tracked_labels = da.from_zarr(output_root["tracked_labels"]).compute()

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
    parser = argparse.ArgumentParser(description="Segment and track rocks in 3D images.")
    parser.add_argument("zarr_path", type=str, help="Path to zarr dataset.")
    parser.add_argument("--thresh_method", "-t", type=str, default="otsu", help="Thresholding method to use. Defaults to 'otsu'.")
    parser.add_argument("--z_range", "-z", nargs=2, type=int, default=None, help="Optional z range to process.")
    parser.add_argument("--boundary_fraction_threshold", "-b", type=float, default=0.15, help="Boundary fraction threshold for merging regions. Defaults to 0.15.")
    args = parser.parse_args()

    __main__(args)
