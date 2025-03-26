import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog
import dictyviz as dv
from dictyviz import channel

sys.path.insert(0,'Y:\\jennifer')
cwd = os.getcwd()
#zarrFile = cwd + "/" + sys.argv[1]

# select zarr file
Tk().withdraw() 
zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
os.chdir(zarrFile)
outputFile = zarrFile + '\\makeOrthoMaxProjMovies_out.txt'
print(zarrFile)
with open(outputFile, 'w') as f:
    print('Zarr file:', zarrFile, '\n', file=f)

    voxelDims = dv.getVoxelDimsFromXML(zarrFile)

    # define channels
    cells_red = channel(name='cells_red', nChannel=0, voxelDims=voxelDims, scaleMax=400)
    cells_green = channel(name='cells_green', nChannel=1, voxelDims=voxelDims, scaleMax=1200)

    # create root store and analysis group
    dv.createRootStore(zarrFile)
    root = zarr.open(zarrFile, mode='r+')
    dv.createZarrGroup(root, 'analysis')
    print('Root store created at ', datetime.datetime.now(), file=f)

    # calculate max projections
    dv.calcMaxProjections(root, res_lvl=0)
    dv.calcSlicedMaxProjections(root, res_lvl=0)
    print('Max projections calculated at ', datetime.datetime.now(), file=f)
