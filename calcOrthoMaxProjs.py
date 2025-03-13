import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog
import dictyviz as dv

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
    dv.calcSlicedMaxProjections(root, res_lvl=0)
    print('Max projections calculated at ', datetime.datetime.now(), file=f)
