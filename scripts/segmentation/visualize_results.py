import os
import napari
import zarr
import dask.array as da
import numpy as np
import pandas as pd
from pathlib import Path

def main(segmentation_path):
    # Load the segmentation data

    parent_path = segmentation_path.parent
    print(f"Parent path: {parent_path}")
    parent_root = zarr.open(parent_path, mode='r')
    seg_root = zarr.open(segmentation_path, mode='r')
    segmented_rocks = da.from_zarr(seg_root['segmented_rocks'])
    segmented_rocks_merge = da.from_zarr(seg_root['segmented_rocks_merge'])
    rock_images = da.from_zarr(parent_root['shadows_removed'])
    tracked_labels = da.from_zarr(seg_root['tracked_labels'])
    # Find the directory ending in .zarr
    parent_parent_path = parent_path.parent
    parent_zarr_dir = [d for d in parent_parent_path.iterdir() if d.is_dir() and d.suffix == '.zarr']
    parent_zarr_path = parent_zarr_dir[0] / '0' / '0' if parent_zarr_dir else None
    if not parent_zarr_dir:
        print(f"No .zarr directory found in {parent_parent_path}")
        cell_images = None
    else:
        print(f".zarr directory found: {parent_zarr_dir[0]}")
        cell_images = da.from_zarr(parent_zarr_path, mode='r')
        cell_images = cell_images[:, 0, :, :, :]  # assuming channel 0 is cells
        print(f"Raw images shape: {cell_images.shape}, dtype: {cell_images.dtype}")

    # crop = (slice(65, 110), slice(0, 78), slice(990, 1390), slice(465, 865))
    # if cell_images is not None:
    #     cell_images = cell_images[crop[0], 0:88, crop[2], crop[3]]
    #     print(f"Cell images shape: {cell_images.shape}, dtype: {cell_images.dtype}")
    # rock_images = rock_images[crop[0], crop[1], crop[2], crop[3]]
    # segmented_rocks = segmented_rocks[crop[0], crop[1], crop[2], crop[3]]
    # segmented_rocks_merge = segmented_rocks_merge[crop[0], crop[1], crop[2], crop[3]]
    # tracked_labels = tracked_labels[crop[0], crop[1], crop[2], crop[3]]


    print(f"Rock images shape: {rock_images.shape}, dtype: {rock_images.dtype}")

    tracks_path = segmentation_path / "rock_tracks.csv"

    track_df = pd.read_csv(tracks_path)
    # Create a Napari viewer and add the segmentation data as a label layer
    viewer = napari.Viewer()
    if cell_images is not None:
        viewer.add_image(cell_images, name='Cells Channel',  colormap='gray_r')
    viewer.add_image(rock_images, name='Rocks Channel', colormap='gray_r')
    viewer.add_labels(segmented_rocks, name='Segmented Rocks')
    viewer.add_labels(segmented_rocks_merge, name='Segmented Rocks Merge')
    viewer.add_labels(tracked_labels, name='Tracked Labels')
    viewer.add_tracks(track_df[["track_id", "frame", "centroid-0", "centroid-1", "centroid-2"]], name='Tracks')

    # Start the Napari event loop
    napari.run()

if __name__ == "__main__":
    # segmentation_path = Path("Y:\\jennifer\\cryolite\\cryolite_mixin_test65_2024-04-16\\NC28.1_overnight_day1\\segmentation\\4_70_otsu_thresh_0.1_boundary_frac\\")
    segmentation_path = Path("/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test45_2024-01-30/segmentation/10_70_otsu_thresh_0.08_boundary_frac")
    main(segmentation_path)