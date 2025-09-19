import os
import napari
import zarr

def main(data_path):

    # Load the Zarr file
    zarr_data = zarr.open(data_path, mode='r')

    # Initialize Napari viewer
    viewer = napari.Viewer()

    # Add each array in the Zarr group as a separate layer
    array = zarr_data['0']['0']
    # Invert order rock channel
    cells = array[:,0]
    rocks = array[:,1]
    rocks = 65535 - rocks
    rocks[rocks == 65535] = 0  # set background back to 0

    viewer.add_image(cells, name="Cells Channel", colormap='bop purple', scale=(1, -1, 1, 1))
    viewer.add_image(rocks, name="Rocks Channel", colormap='gray', scale=(1, -1, 1, 1))

    # Start the Napari event loop
    napari.run()

if __name__ == "__main__":
    # Define the path to the Zarr file
    data_path = "/groups/sgro/sgrolab/jennifer/cryolite/cryolite_mixin_test25_2023-05-17/2023-05-17_ERH_mixin25_overnight_restart_Emily Green FarRed_2_F1.zarr"
    
    # Run the main function
    main(data_path)