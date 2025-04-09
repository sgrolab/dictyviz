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
    
    outputFile = zarrFile + '/makeOrthoMaxProjMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)

        # create root store
        dv.createRootStore(zarrFile)
        root = zarr.open(zarrFile, mode='r+')

        # define channels
        channels = dv.getChannelsFromJSON(zarrFile+'/parameters.json')
        for channel in channels:
            channel.voxelDims = dv.getVoxelDimsFromXML(zarrFile+'/OME/METADATA.ome.xml')
            print("Channel " + channel.name + ": Min = " + str(channel.scaleMin) + ", Max = " + str(channel.scaleMax))

        # create movies group
        dv.createZarrGroup(root, 'movies')
        os.chdir(zarrFile + '/movies')

        # create dask client
        try:
            client = Client(threads_per_worker=8, n_workers=1)
            print('Dask client created at ', datetime.datetime.now())
        except:
            print('Dask client could not be created')
            sys.exit() 

        # # create ortho max videos
        # for channel in channels:
        #     dv.makeOrthoMaxVideo(root, channel)
        #     dv.makeSlicedOrthoMaxVideos(root, channel)
        #     dv.makeZDepthOrthoMaxVideo(root, channel, 'gist_rainbow_r')
        # dv.makeCompOrthoMaxVideo(root, channels)
        # print('Ortho max videos created at ', datetime.datetime.now()) 

        # submit movie tasks
        try:
            wait([client.submit(dv.makeOrthoMaxVideo, root, channel[0]),
                client.submit(dv.makeOrthoMaxVideo, root, channel[1]),
                client.submit(dv.makeCompOrthoMaxVideo, root, channels),
                client.submit(dv.makeSlicedOrthoMaxVideos, root, channel[0]),
                client.submit(dv.makeSlicedOrthoMaxVideos, root, channel[1]),
                client.submit(dv.makeZDepthOrthoMaxVideo, root, channel[0], 'gist_rainbow_r'),
                client.submit(dv.makeZDepthOrthoMaxVideo, root, channel[1], 'gist_rainbow_r'),])
            print('Ortho max videos created at ', datetime.datetime.now())   
        except:
            print('Dask tasks could not be submitted')
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
