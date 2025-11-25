import os
import napari
import zarr
import numpy as np
import pandas as pd

def main(segmentation_path):
    # Load the segmentation data

    seg_root = zarr.open(segmentation_path, mode='r')
    tracked_labels = seg_root['tracked_labels'][0:3]
    segmented_rocks = seg_root['segmented_rocks'][0:3]
    segmented_rocks_merge = seg_root['segmented_rocks_merge'][0:3]
    rock_images = seg_root['shadows_removed'][0:3]

    # tracked_labels_path = os.path.join(segmentation_path, "tracked_labels")
    # tracked_labels = zarr.open(tracked_labels_path, mode='r')

    # Invert rock channel
    print(f"Rock images shape: {rock_images.shape}, dtype: {rock_images.dtype}")
    rock_images = 65535 - np.array(rock_images)
    rock_images[rock_images == 65535] = 0  # set background back to 0

    tracks_path = segmentation_path + "/rock_tracks.csv"

    track_df = pd.read_csv(tracks_path)
    # Create a Napari viewer and add the segmentation data as a label layer
    viewer = napari.Viewer()
    viewer.add_image(rock_images, name='Rocks Channel', colormap='gray')
    viewer.add_labels(segmented_rocks, name='Segmented Rocks')
    viewer.add_labels(segmented_rocks_merge, name='Segmented Rocks Merge')
    viewer.add_labels(tracked_labels, name='Tracked Labels', opacity=0.5)
    #viewer.add_tracks(track_df[["track_id", "frame", "centroid-0", "centroid-1", "centroid-2"]], name='Tracks')

    # Start the Napari event loop
    napari.run()

if __name__ == "__main__":
    segmentation_path = "Y:\\jennifer\\cryolite\\cryolite_mixin_test65_2024-04-16\\NC28.1_overnight_day1\\segmentation\\4_70\\"
    #segmentation_path = "/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test65_2024-04-16/NC28.1_overnight_day1/segmentation/4_70/"
    main(segmentation_path)