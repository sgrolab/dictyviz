import sys
import os
import datetime
import zarr
import json
from tkinter import Tk, filedialog
from dask.distributed import Client, wait

# Add src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.append(src_path)

import dictyviz as dv
from dictyviz import channel

# TODO: add cropID as kwarg
def main(zarrFile=None, cropID=None):
    if zarrFile is None:
        # select zarr file
        Tk().withdraw() 
        zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    if cropID is None:
        cropID = ''
    print(zarrFile)
    print('Crop ID:', cropID)
    
    # create movies directory in the parent folder of the zarr file
    parentDir = os.path.dirname(zarrFile)
    moviesDir = os.path.join(parentDir, 'movies')
    if not os.path.exists(moviesDir):
        os.makedirs(moviesDir)
    # if cropping, create a subdirectory for the cropID
    if cropID != '':
        moviesDir = os.path.join(moviesDir, 'movies'+cropID)
        if not os.path.exists(moviesDir):
            os.makedirs(moviesDir)
    os.chdir(moviesDir)

    outputFile = parentDir + '/makeOrthoMaxProjMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)
        print('Movies directory:', moviesDir, '\n', file=f)

        # open analysis zarr file
        maxProjectionsRoot = zarr.open(parentDir + '/analysis/max_projections' + cropID, mode='r+')
        slicedMaxProjectionsRoot = zarr.open(parentDir + '/analysis/sliced_max_projections' + cropID, mode='r+')

        # define channels
        channels = dv.getChannelsFromJSON(parentDir+'/parameters.json')
        for channel in channels:
            channel.voxelDims = dv.getVoxelDimsFromXML(zarrFile+'/OME/METADATA.ome.xml')
            print("Channel " + channel.name + ": Min = " + str(channel.scaleMin) + ", Max = " + str(channel.scaleMax), file=f)

        # define colormaps
        with open(parentDir+'/parameters.json') as json_file:
            colormaps = json.load(json_file)['movieSpecs']
            primaryColormap = colormaps['primaryColormap']
            zDepthColormap = colormaps['zDepthColormap']

        # create dask client
        if len(channels) == 1:
            nTasks = 5
        else:
            nTasks = (5*len(channels)) + 1

        try:
            client = Client(threads_per_worker=nTasks, n_workers=1)
            print('\nDask client created at ', datetime.datetime.now(), file=f)
        except:
            print('\nDask client could not be created', file=f)
            sys.exit() 

        #submit movie tasks
        try:
            if len(channels) == 1:
                wait([client.submit(dv.makeOrthoMaxVideoClean, maxProjectionsRoot, channels[0], primaryColormap),
                       client.submit(dv.makeOrthoMaxVideo, maxProjectionsRoot, channels[0], primaryColormap),
                       client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channels[0], 'x', primaryColormap),
                       client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channels[0], 'y', primaryColormap),
                       client.submit(dv.makeZDepthOrthoMaxVideo, maxProjectionsRoot, channels[0], zDepthColormap)])
            else:
                for channel in channels:
                    wait([client.submit(dv.makeOrthoMaxVideoClean, maxProjectionsRoot, channel, primaryColormap),
                        client.submit(dv.makeOrthoMaxVideo, maxProjectionsRoot, channel, primaryColormap),
                        client.submit(dv.makeCompOrthoMaxVideo, maxProjectionsRoot, channels),
                        client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channel, 'x', primaryColormap),
                        client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channel, 'y', primaryColormap),
                        client.submit(dv.makeZDepthOrthoMaxVideo, maxProjectionsRoot, channel, zDepthColormap)])
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
        zarrFile = sys.argv[1]
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
        if len(sys.argv) > 2:
            cropID = sys.argv[2]
    else:
        zarrFile = None
        cropID = None
    main(zarrFile, cropID)
