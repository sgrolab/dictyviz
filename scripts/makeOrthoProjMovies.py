import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog

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
    print(zarrFile)
    
    outputFile = zarrFile + '\\makeOrthoMaxProjMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)

        # create root store and movies group
        dv.createRootStore(zarrFile)
        root = zarr.open(zarrFile, mode='r+')

        voxelDims = dv.getVoxelDimsFromXML(zarrFile+'/OME/METADATA.ome.xml')

        # define channels
        channels = []
        channels.append(channel(name='cells', nChannel=1, voxelDims=voxelDims, scaleMax=1800, adjMin=200))
        channels.append(channel(name='rocks', nChannel=0, voxelDims=voxelDims, scaleMax=3000, adjMin=1500))
        for channel in channels:
            print("Channel " + channel.name + ": Min = " + str(channel.adjMin) + ", Max = " + str(channel.scaleMax))

        dv.createZarrGroup(root, 'movies')
        os.chdir(zarrFile + '/movies')

        for channel in channels:
            dv.makeOrthoMaxVideo(root, channel)
            dv.makeSlicedOrthoMaxVideos(root, channel)
            dv.makeZDepthOrthoMaxVideo(root, channel, 'gist_rainbow_r')
        dv.makeCompOrthoMaxVideo(root, channels)
        print('Ortho max videos created at ', datetime.datetime.now()) 

if __name__ == '__main__':
    zarrFile = sys.argv[1] if len(sys.argv) > 1 else None
    main(zarrFile)
