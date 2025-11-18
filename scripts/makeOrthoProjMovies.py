import sys
import os
import datetime
import zarr
import json
from tkinter import Tk, filedialog
from dask.distributed import Client, wait

import dictyviz as dv
from dictyviz import Channel
from dictyviz.utils import get_channels, get_voxel_dims

# TODO: add crop_id as kwarg
def main(zarr_file=None, crop_id=None):
    if zarr_file is None:
        # select zarr file
        Tk().withdraw() 
        zarr_file = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
        if not os.path.isdir(zarr_file):
            print(f"Error: The provided path '{zarr_file}' is not a valid directory.")
            sys.exit(1)
    if crop_id is None:
        crop_id = ''
    print(zarr_file)
    print('Crop ID:', crop_id)
    
    # create movies directory in the parent folder of the zarr file
    parent_dir = os.path.dirname(zarr_file)
    movies_dir = os.path.join(parent_dir, 'movies')
    if not os.path.exists(movies_dir):
        os.makedirs(movies_dir)
    # if cropping, create a subdirectory for the crop_id
    if crop_id != '':
        movies_dir = os.path.join(movies_dir, 'movies'+crop_id)
        if not os.path.exists(movies_dir):
            os.makedirs(movies_dir)
    os.chdir(movies_dir)

    output_file = parent_dir + '/makeOrthoMaxProjMovies_out.txt'
    with open(output_file, 'w') as f:
        print('Zarr file:', zarr_file, '\n', file=f)
        print('Movies directory:', movies_dir, '\n', file=f)

        # open analysis zarr file
        max_projections_root = zarr.open(parent_dir + '/analysis/max_projections' + crop_id, mode='r+')
        sliced_max_projections_root = zarr.open(parent_dir + '/analysis/sliced_max_projections' + crop_id, mode='r+')

        # define channels
        channels = get_channels(parent_dir+'/parameters.json')
        for channel in channels:
            channel.voxel_dims = get_voxel_dims(zarr_file+'/OME/METADATA.ome.xml')
            print("Channel " + channel.name + ": Min = " + str(channel.scale_min) + ", Max = " + str(channel.scale_max), file=f)

        # define colormaps
        with open(parent_dir+'/parameters.json') as json_file:
            colormaps = json.load(json_file)['movieSpecs']
            primary_colormap = colormaps['primaryColormap']
            z_depth_colormap = colormaps['zDepthColormap']

        # create dask client
        if len(channels) == 1:
            n_tasks = 5
        else:
            n_tasks = (5*len(channels)) + 1

        try:
            client = Client(threads_per_worker=n_tasks, n_workers=1)
            print('\nDask client created at ', datetime.datetime.now(), file=f)
        except:
            print('\nDask client could not be created', file=f)
            sys.exit() 

        #submit movie tasks
        try:
            if len(channels) == 1:
                wait([client.submit(dv.make_ortho_max_video_clean, max_projections_root, channels[0], primary_colormap),
                       client.submit(dv.make_ortho_max_video, max_projections_root, channels[0], primary_colormap),
                       client.submit(dv.make_sliced_ortho_max_videos, sliced_max_projections_root, channels[0], 'x', primary_colormap),
                       client.submit(dv.make_sliced_ortho_max_videos, sliced_max_projections_root, channels[0], 'y', primary_colormap),
                       client.submit(dv.make_z_depth_ortho_max_video, max_projections_root, channels[0], z_depth_colormap)])
            else:
                for channel in channels:
                    wait([client.submit(dv.make_ortho_max_video_clean, max_projections_root, channel, primary_colormap),
                        client.submit(dv.make_ortho_max_video, max_projections_root, channel, primary_colormap),
                        client.submit(dv.make_comp_ortho_max_video, max_projections_root, channels),
                        client.submit(dv.make_sliced_ortho_max_videos, sliced_max_projections_root, channel, 'x', primary_colormap),
                        client.submit(dv.make_sliced_ortho_max_videos, sliced_max_projections_root, channel, 'y', primary_colormap),
                        client.submit(dv.make_z_depth_ortho_max_video, max_projections_root, channel, z_depth_colormap)])
            print('Ortho max videos created at ', datetime.datetime.now(), file=f)   
        except Exception as e:
            print('Dask tasks could not be submitted', file=f)
            import traceback
            traceback.print_exc()
            client.shutdown()
            sys.exit()

        client.shutdown()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        zarr_file = sys.argv[1]
        if not os.path.isdir(zarr_file):
            print(f"Error: The provided path '{zarr_file}' is not a valid directory.")
            sys.exit(1)
        if len(sys.argv) > 2:
            crop_id = sys.argv[2]
    else:
        zarr_file = None
        crop_id = None
    main(zarr_file, crop_id)
