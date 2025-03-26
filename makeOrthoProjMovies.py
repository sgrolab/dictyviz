import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog
import dictyviz as dv
from dictyviz import channel

def main(zarrFile=None):
    if zarrFile is None:
        # select zarr file
        Tk().withdraw() 
        zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')

    # create root store and movies group
    dv.createRootStore(zarrFile)
    root = zarr.open(zarrFile, mode='r+')

    voxelDims = dv.getVoxelDimsFromXML(zarrFile+'/OME/METADATA.ome.xml')

    # define channels
    cells = channel(name='cells', nChannel=1, voxelDims=voxelDims, scaleMax=2300)
    rocks = channel(name='rocks', nChannel=0, voxelDims=voxelDims, scaleMax=3000, adjMin=1000)

    dv.createZarrGroup(root, 'movies')
    os.chdir(zarrFile + '/movies')

    dv.makeOrthoMaxVideo(root, cells)
    dv.makeOrthoMaxVideo(root, rocks)
    dv.makeCompOrthoMaxVideo(root, [cells, rocks])
    dv.makeSlicedOrthoMaxVideos(root, cells)
    dv.makeSlicedOrthoMaxVideos(root, rocks)
    dv.makeZDepthOrthoMaxVideo(root, cells, 'gist_rainbow_r')
    dv.makeZDepthOrthoMaxVideo(root, rocks, 'gist_rainbow_r')
    print('Ortho max videos created at ', datetime.datetime.now()) 

if __name__ == '__main__':
    zarrFile = sys.argv[1] if len(sys.argv) > 1 else None
    main(zarrFile) 
