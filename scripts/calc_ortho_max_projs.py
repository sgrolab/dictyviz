import sys
import os
import datetime
import zarr
from tkinter import Tk, filedialog

import dictyviz as dv
from utils import create_root_store

def main(zarr_file=None, crop_id=None):
    if zarr_file is None:
        # select zarr file
        Tk().withdraw() 
        zarr_file = filedialog.askdirectory(initialdir='cryolite', title='Select a zarr file')
        if not os.path.isdir(zarr_file):
            print(f"Error: The provided path '{zarr_file}' is not a valid directory.")
            sys.exit(1)
    print(zarr_file)
    print('Crop ID:', crop_id)

    # get parent directory of zarr file
    parent_dir = os.path.dirname(zarr_file)
    os.chdir(parent_dir)

    output_file = parent_dir + '/calcOrthoMaxProjs_out.txt'
    with open(output_file, 'w') as f:
        print('Zarr file:', zarr_file, '\n', file=f)
        if crop_id is None:
            crop_id = ''
        else:
            print('Crop ID:', crop_id, '\n', file=f)

        # create root stores and analysis group
        create_root_store(zarr_file)
        root = zarr.open(zarr_file, mode='r+')
        
        if 'analysis' not in parent_dir:
            os.makedirs(parent_dir + '/analysis', exist_ok=True)
        analysis_dir = parent_dir + '/analysis'
        print('Analysis directory:', analysis_dir, file=f)

        # check if projections have already been calculated
        if os.path.isdir(os.path.join(analysis_dir, 'max_projections' + crop_id)):
            print('Max projections already calculated, skipping calculation.', file=f)
            return

        # calculate max projections
        create_root_store(parent_dir + '/analysis/max_projections' + crop_id)
        max_projections_root = zarr.open(parent_dir + '/analysis/max_projections' + crop_id, mode='r+')
        print('Root store created at ', datetime.datetime.now(), file=f)
        dv.calc_max_projections(root, max_projections_root, res_lvl=0)
        print('Max projections calculated at ', datetime.datetime.now(), file=f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        zarr_file = sys.argv[1]
        if not os.path.isdir(zarr_file):
            print(f"Error: The provided path '{zarr_file}' is not a valid directory.")
            sys.exit(1)
        if len(sys.argv) > 2:
            crop_id = sys.argv[2]
    else:
        zarr_file = None
        crop_id = None
    main(zarr_file, crop_id)
