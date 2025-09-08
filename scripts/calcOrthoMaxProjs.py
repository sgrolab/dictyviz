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

def main(zarrFile=None, cropID=None):
    if zarrFile is None:
        # select zarr file
        Tk().withdraw() 
        zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select a zarr file')
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    print(zarrFile)
    print('Crop ID:', cropID)

    # get parent directory of zarr file
    parentDir = os.path.dirname(zarrFile)
    os.chdir(parentDir)

    outputFile = parentDir + '/calcOrthoMaxProjs_out.txt'
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)
        if cropID is None:
            cropID = ''
        else:
            print('Crop ID:', cropID, '\n', file=f)

        # create root stores and analysis group
        dv.createRootStore(zarrFile)
        root = zarr.open(zarrFile, mode='r+')
        
        if 'analysis' not in parentDir:
            os.makedirs(parentDir + '/analysis', exist_ok=True)
        analysisDir = parentDir + '/analysis'
        print('Analysis directory:', analysisDir, file=f)

        # check if projections have already been calculated
        if os.path.isdir(os.path.join(analysisDir, 'max_projections' + cropID)):
            print('Max projections already calculated, skipping calculation.', file=f)
            return

        # calculate max projections
        dv.createRootStore(parentDir + '/analysis/max_projections' + cropID)
        maxProjectionsRoot = zarr.open(parentDir + '/analysis/max_projections' + cropID, mode='r+')
        print('Root store created at ', datetime.datetime.now(), file=f)
        dv.calcMaxProjections(root, maxProjectionsRoot, res_lvl=0)
        print('Max projections calculated at ', datetime.datetime.now(), file=f)

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
