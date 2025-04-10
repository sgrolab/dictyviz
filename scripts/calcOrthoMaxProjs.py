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

def main(zarrFile=None):
    if zarrFile is None:
        # select zarr file
        Tk().withdraw() 
        zarrFile = filedialog.askdirectory(initialdir='cryolite', title='Select a zarr file')
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    print(zarrFile)

    os.chdir(zarrFile)
    outputFile = zarrFile + '/calcOrthoMaxProjs_out.txt'
    print(zarrFile)
    with open(outputFile, 'w') as f:
        print('Zarr file:', zarrFile, '\n', file=f)

        # create root store and analysis group
        dv.createRootStore(zarrFile)
        root = zarr.open(zarrFile, mode='r+')
        dv.createZarrGroup(root, 'analysis')
        print('Root store created at ', datetime.datetime.now(), file=f)

        # check if projections have already been calculated
        if root['analysis']['max_projections']['maxx']:
            print('Max projections already calculated, skipping calculation.', file=f)
            return

        # calculate max projections
        dv.calcMaxProjections(root, res_lvl=0)
        print('Max projections calculated at ', datetime.datetime.now(), file=f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        zarrFile = sys.argv[1]
        if not os.path.isdir(zarrFile):
            print(f"Error: The provided path '{zarrFile}' is not a valid directory.")
            sys.exit(1)
    else:
        zarrFile = None
    main(zarrFile)
