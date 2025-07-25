import sys
import os
import datetime
import zarr
import numpy as np
from tqdm import tqdm
import json
from tkinter import Tk, filedialog

# Add src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', '..', 'src')
sys.path.append(src_path)

import dictyviz as dv

def main(opticalFolder=None, cropID=None):
    if opticalFolder is None:
        # select optical results folder
        Tk().withdraw() 
        opticalFolder = filedialog.askdirectory(initialdir='cryolite', title='Select an optical results folder')  
        if not os.path.isdir(opticalFolder):  
            print(f"Error: The provided path '{opticalFolder}' is not a valid directory.")
            sys.exit(1)
    
    print(opticalFolder)
    print('Crop ID:', cropID)
    
    # Handle cropID being None
    if cropID is None:
        cropID = ''

    # get parent directory of optical folder
    parentDir = os.path.dirname(opticalFolder)
    os.chdir(parentDir)

    outputFile = parentDir + '/calcOrthoFlowMaxProjs_out.txt'
    with open(outputFile, 'w') as f:
        print('Optical folder:', opticalFolder, file=f)  
        print('Crop ID:', cropID, file=f)  

        # DON'T create zarr store from opticalFolder - it's just a directory with frame results
        # opticalFolder contains frame directories with .npy files, not zarr data
    
        if not os.path.exists(parentDir + '/analysis'): 
            os.makedirs(parentDir + '/analysis', exist_ok=True)
        
        analysisDir = parentDir + '/analysis'
        print('Analysis directory:', analysisDir, file=f)

        # check if projections have already been calculated
        flow_proj_dir = os.path.join(analysisDir, 'optical_flow_max_projections' + cropID)
        if os.path.isdir(flow_proj_dir):
            print('Optical flow max projections already calculated, skipping calculation.', file=f)
            return

        # calculate max projections - create NEW zarr store for results
        dv.createRootStore(flow_proj_dir)
        maxProjectionsRoot = zarr.open(flow_proj_dir, mode='r+')
        print('Root store created at ', datetime.datetime.now(), file=f)
        
        # Pass the opticalFolder to the function so it knows where to find the .npy files
        try:
            dv.calcOpticalFlowMaxProjections(maxProjectionsRoot, opticalFolder, cropID)
            print('Optical flow max projections calculated at ', datetime.datetime.now(), file=f)
        except Exception as e:
            print(f'ERROR in calcOpticalFlowMaxProjections: {e}', file=f)
            import traceback
            traceback.print_exc(file=f)
            raise

if __name__ == '__main__':
    if len(sys.argv) > 1:
        opticalFolder = sys.argv[1]  
        if not os.path.isdir(opticalFolder): 
            print(f"Error: The provided path '{opticalFolder}' is not a valid directory.")
            sys.exit(1)
        if len(sys.argv) > 2:
            cropID = sys.argv[2]
        else:
            cropID = None 
    else:
        opticalFolder = None 
        cropID = None
    main(opticalFolder, cropID)