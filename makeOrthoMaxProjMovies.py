import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog
import dictyviz as dv

sys.path.insert(0,'Y:\\jennifer')

# select zarr file
Tk().withdraw() 
zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
outputFile = zarrFile + '\\makeOrthoMaxProjMovies_out.txt'
print(zarrFile)
with open(outputFile, 'w') as f:
    print('Zarr file:', zarrFile, '\n', file=f)

    # define channels
    cells = 0
    rocks = 1

    # create root store and analysis group
    dv.createRootStore(zarrFile)
    root = zarr.open(zarrFile, mode='r+')
    dv.createZarrGroup(root, 'analysis')
    print('Root store created at ', datetime.datetime.now(), file=f)

    # calculate max projections
    dv.calcMaxProjections(root, res_lvl=0)
    print('Max projections calculated at ', datetime.datetime.now(), file=f)

    # create movies group
    dv.createZarrGroup(root, 'movies')
    os.chdir(zarrFile + '\\movies')
    scaleMax = 700

    # make max projection videos
    dv.makeOrthoMaxVideo('cells_orthomax3.avi', root, cells, scaleMax)
    dv.makeOrthoMaxVideo('rocks_orthomax3.avi', root, rocks, scaleMax)
    print('OrthoMax videos created at ', datetime.datetime.now(), '\n', file=f)