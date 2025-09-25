import os
import napari
import tifffile as tiff
import numpy as np
import pandas as pd

def main(segmentation_path):
    # Load the segmentation data
    tracked_labels_path = os.path.join(segmentation_path, "tracked_labels.tiff")
    tracked_labels = tiff.imread(tracked_labels_path)

    rock_images = []
    for t in range(tracked_labels.shape[0]):
        tp_output_dir = segmentation_path + f"t{t}/"
        rocks_img_path = tp_output_dir + "cell_shadows_removed_98thresh_3medfilt.tif"
        rock_images.append(tiff.imread(rocks_img_path))
    rock_images = np.array(rock_images)
    # Invert rock channel
    print(f"Rock images dtype: {rock_images.dtype}, min: {rock_images.min()}, max: {rock_images.max()}")
    rock_images = 65535 - rock_images
    rock_images[rock_images == 65535] = 0  # set background back to 0

    tracks_path = segmentation_path + "rock_tracks.csv"

    track_df = pd.read_csv(tracks_path)
    # Create a Napari viewer and add the segmentation data as a label layer
    viewer = napari.Viewer()
    viewer.add_image(rock_images, name='Rocks Channel', colormap='gray')
    viewer.add_labels(tracked_labels, name='Tracked Labels', opacity=0.5)
    viewer.add_tracks(track_df[["track_id", "frame", "centroid-0", "centroid-1", "centroid-2"]], name='Tracks')

    # Start the Napari event loop
    napari.run()

if __name__ == "__main__":
    #segmentation_path = "Y:\\jennifer\\cryolite\\cryolite_mixin_test65_2024-04-16\\WS205_overnight_day2\\2024-04-17_ERH_mixin65_plate2_WS205_overnight_day2_ERH Red FarRed_crop2\\segmentation\\"
    segmentation_path = "/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test65_2024-04-16/WS205_overnight_day2/2024-04-17_ERH_mixin65_plate2_WS205_overnight_day2_ERH Red FarRed/segmentation/"
    main(segmentation_path)