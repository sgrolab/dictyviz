import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog
from dask.distributed import Client, wait

# Add src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.append(src_path)

import dictyviz as dv
from dictyviz import channel

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

    outputFile = parentDir + '/makeOrthoMaxProjMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)

        # open analysis zarr file
        analysisRoot = zarr.open(parentDir, mode='r+')

        # define channels
        channels = dv.getChannelsFromJSON(parentDir+'/parameters.json')
        for channel in channels:
            channel.voxelDims = dv.getVoxelDimsFromXML(zarrFile+'/OME/METADATA.ome.xml')
            print("Channel " + channel.name + ": Min = " + str(channel.scaleMin) + ", Max = " + str(channel.scaleMax), file=f)

        # # create dask client
        try:
            client = Client(threads_per_worker=8, n_workers=1)
            print('\nDask client created at ', datetime.datetime.now(), file=f)
        except:
            print('\nDask client could not be created', file=f)
            sys.exit() 

        #submit movie tasks
        try:
            wait([client.submit(dv.makeOrthoMaxVideo, analysisRoot, channels[0]),
                client.submit(dv.makeOrthoMaxVideo, analysisRoot, channels[1]),
                client.submit(dv.makeCompOrthoMaxVideo, analysisRoot, channels),
                client.submit(dv.makeSlicedOrthoMaxVideos, analysisRoot, channels[0]),
                client.submit(dv.makeSlicedOrthoMaxVideos, analysisRoot, channels[1]),
                client.submit(dv.makeZDepthOrthoMaxVideo, analysisRoot, channels[0], 'gist_rainbow_r'),
                client.submit(dv.makeZDepthOrthoMaxVideo, analysisRoot, channels[1], 'gist_rainbow_r'),])
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
