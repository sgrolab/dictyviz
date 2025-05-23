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
def main(zarrFile=None):
    if zarrFile is None:
        # select zarr file
        Tk().withdraw() 
        zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    print(zarrFile)
    
    # create movies directory in the parent folder of the zarr file
    parentDir = os.path.dirname(zarrFile)
    moviesDir = os.path.join(parentDir, 'movies')
    if not os.path.exists(moviesDir):
        os.makedirs(moviesDir)
    os.chdir(moviesDir)
    # TODO: if cropping, create a subdirectory for the cropID
    # TODO: chdir to the cropID subdirectory

    outputFile = parentDir + '/makeOrthoMaxProjMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)
        # TODO: print cropID

        # open analysis zarr file
        # TODO: refactor to use the actual max_projections folder as root
        #analysisRoot = zarr.open(parentDir, mode='r+')
        maxProjectionsRoot = zarr.open(parentDir + '/analysis/max_projections', mode='r+')
        slicedMaxProjectionsRoot = zarr.open(parentDir + '/analysis/sliced_max_projections', mode='r+')

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

        # # create dask client
        try:
            client = Client(threads_per_worker=8, n_workers=1)
            print('\nDask client created at ', datetime.datetime.now(), file=f)
        except:
            print('\nDask client could not be created', file=f)
            sys.exit() 

        #submit movie tasks
        # TODO: switch analysisRoot to max_projections root
        try:
            wait([client.submit(dv.makeOrthoMaxVideo, maxProjectionsRoot, channels[0], primaryColormap),
                client.submit(dv.makeOrthoMaxVideo, maxProjectionsRoot, channels[1], primaryColormap),
                client.submit(dv.makeCompOrthoMaxVideo, maxProjectionsRoot, channels),
                client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channels[0], primaryColormap),
                client.submit(dv.makeSlicedOrthoMaxVideos, slicedMaxProjectionsRoot, channels[1], primaryColormap),
                client.submit(dv.makeZDepthOrthoMaxVideo, maxProjectionsRoot, channels[0], zDepthColormap),
                client.submit(dv.makeZDepthOrthoMaxVideo, maxProjectionsRoot, channels[1], zDepthColormap)])
            print('Ortho max videos created at ', datetime.datetime.now(), file=f)   
        except:
            print('Dask tasks could not be submitted', file=f)
            client.shutdown()
            sys.exit()

        client.shutdown()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        zarrFile = sys.argv[1]
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    else:
        zarrFile = None
    main(zarrFile)
