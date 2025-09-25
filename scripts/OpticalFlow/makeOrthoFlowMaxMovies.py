import sys
import os
import datetime
import zarr
import json
from tkinter import Tk, filedialog
from dask.distributed import Client, wait

# Add src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', '..', 'src')
sys.path.append(src_path)

import dictyviz as dv
from dictyviz import channel

def main(opticalFlowProjectionsDir=None, cropID=None):
    if opticalFlowProjectionsDir is None:
        # select optical flow max projections directory
        Tk().withdraw() 
        opticalFlowProjectionsDir = filedialog.askdirectory(initialdir='cryolite', title='Select optical flow max projections directory')
        if not os.path.isdir(opticalFlowProjectionsDir):
            print(f"Error: The provided path '{opticalFlowProjectionsDir}' is not a valid directory.")
            sys.exit(1)
    if cropID is None:
        cropID = ''
    print(opticalFlowProjectionsDir)
    print('Crop ID:', cropID)
    
    # create movies directory in the parent folder of the optical flow projections
    parentDir = os.path.dirname(opticalFlowProjectionsDir)
    moviesDir = os.path.join(parentDir, 'movies')
    if not os.path.exists(moviesDir):
        os.makedirs(moviesDir)
    # if cropping, create a subdirectory for the cropID
    if cropID != '':
        moviesDir = os.path.join(moviesDir, 'movies'+cropID)
        if not os.path.exists(moviesDir):
            os.makedirs(moviesDir)
    os.chdir(moviesDir)

    outputFile = parentDir + '/makeOrthoFlowMaxMovies_out.txt'
    with open(outputFile, 'w') as f:
        print('Optical flow projections directory:', opticalFlowProjectionsDir, '\n', file=f)
        print('Movies directory:', moviesDir, '\n', file=f)

        # open optical flow max projections zarr file
        try:
            opticalFlowMaxProjectionsRoot = zarr.open(opticalFlowProjectionsDir, mode='r+')
            print('Optical flow max projections opened successfully', file=f)
        except Exception as e:
            print(f'Error opening optical flow max projections: {e}', file=f)
            sys.exit(1)

        # Get voxel dimensions - look for parameters.json in parent directories
        voxelDims = None
        search_dir = parentDir
        for _ in range(3):  # Search up to 3 levels up
            params_file = os.path.join(search_dir, 'parameters.json')
            if os.path.exists(params_file):
                try:
                    with open(params_file) as json_file:
                        params = json.load(json_file)
                        if 'voxelDims' in params:
                            voxelDims = params['voxelDims']
                            break
                        # Also try to get from imagingParameters if it exists
                        elif 'imagingParameters' in params and 'voxelDims' in params['imagingParameters']:
                            voxelDims = params['imagingParameters']['voxelDims']
                            break
                except Exception as e:
                    print(f'Error reading parameters.json: {e}', file=f)
            search_dir = os.path.dirname(search_dir)
        
        # Default voxel dimensions if not found
        if voxelDims is None:
            voxelDims = [0.1625, 0.1625, 0.5]  # Default values in microns
            print(f'Using default voxel dimensions: {voxelDims}', file=f)
        else:
            print(f'Found voxel dimensions: {voxelDims}', file=f)

        # Create optical flow channel
        opticalFlowChannel = channel(
            name='optical_flow',
            nChannel=0,  # Not used for optical flow
            scaleMax=1.0,  # Optical flow data is already normalized
            scaleMin=0.0,
            gamma=1.0,
            invertChannel=False,
            voxelDims=voxelDims
        )
        
        print("Optical Flow Channel: Min = " + str(opticalFlowChannel.scaleMin) + ", Max = " + str(opticalFlowChannel.scaleMax), file=f)

        # create dask client
        try:
            client = Client(threads_per_worker=11, n_workers=1)
            print('\nDask client created at ', datetime.datetime.now(), file=f)
        except Exception as e:
            print(f'\nDask client could not be created: {e}', file=f)
            sys.exit() 

        # submit optical flow movie task
        try:
            wait([client.submit(dv.makeOrthoMaxOpticalFlowVideo, opticalFlowMaxProjectionsRoot, opticalFlowChannel)])
            print('Optical flow ortho max video created at ', datetime.datetime.now(), file=f)   
        except Exception as e:
            print(f'Dask task could not be submitted: {e}', file=f)
            client.shutdown()
            sys.exit()

        client.shutdown()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        opticalFlowProjectionsDir = sys.argv[1]
        if not os.path.isdir(opticalFlowProjectionsDir):
            print(f"Error: The provided path '{opticalFlowProjectionsDir}' is not a valid directory.")
            sys.exit(1)
        if len(sys.argv) > 2:
            cropID = sys.argv[2]
        else:
            cropID = None
    else:
        opticalFlowProjectionsDir = None
        cropID = None
    main(opticalFlowProjectionsDir, cropID)