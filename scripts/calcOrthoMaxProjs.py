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

sys.path.insert(0,'Y:\\jennifer')
cwd = os.getcwd()
#zarrFile = cwd + "/" + sys.argv[1]

# select zarr file
Tk().withdraw() 
zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select zarr file(s)')
os.chdir(zarrFile)
outputFile = zarrFile + '\\calcOrthoMaxProjs_out.txt'
print(zarrFile)
with open(outputFile, 'w') as f:
    print('Zarr file:', zarrFile, '\n', file=f)

    # create root store and analysis group
    dv.createRootStore(zarrFile)
    root = zarr.open(zarrFile, mode='r+')
    dv.createZarrGroup(root, 'analysis')
    print('Root store created at ', datetime.datetime.now(), file=f)

    # calculate max projections
    dv.calcMaxProjections(root, res_lvl=0)
    dv.calcSlicedMaxProjections(root, res_lvl=0)
    print('Max projections calculated at ', datetime.datetime.now(), file=f)
