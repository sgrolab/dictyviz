# Dicty data functions for ome-zarr datasets

import os, zarr, cv2, cmapy, copy, math
import numpy as np
from tqdm import tqdm

def returnAnalysisGroup(root):
    if 'analysis' in root:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    return analysis_group

def getDimensions(res_array):
    return res_array.shape[0], res_array.shape[2], res_array.shape[3], res_array.shape[4]

def calcMaxProjections(file, z, nChannel, res_lvl=0):

    # define resolution level
    res_array = z[str(res_lvl)]

    # get dataset dimensions
    len_t, len_z, len_y, len_x = getDimensions(res_array)

    # create nested directory structure within analysis group 
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # create analysis group if it does not already exist
    analysis_group = returnAnalysisGroup(root)

    maxProjectionsGroup = analysis_group.create_group('max_projections')

    # create zarr arrays for each max projection 
    maxz = maxProjectionsGroup.zeros('maxz',shape=(len_t,len_y,len_x),chunks=(1,64,64))
    maxx = maxProjectionsGroup.zeros('maxx',shape=(len_t,len_z,len_y),chunks=(1,64,64))
    maxy = maxProjectionsGroup.zeros('maxy',shape=(len_t,len_z,len_x),chunks=(1,64,64))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(len_t)):
        frame = res_array[i, nChannel, :, :, :]
        maxz[i] = np.max(frame,axis=0)
        maxx[i] = np.max(frame,axis=2)
        maxy[i] = np.max(frame,axis=1)

def calcSlicedMaxProjections(file, z, nChannel, res_lvl=0):
    # define resolution level
    res_array = z[str(res_lvl)]

    # get dataset dimensions
    len_t, len_z, len_y, len_x = getDimensions(res_array)

    # create nested directory structure within analysis group 
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # create analysis group if it does not already exist
    analysis_group = returnAnalysisGroup(root)

    slicedMaxProjectionsGroup = analysis_group.create_group('sliced_max_projections')

    # set number of slices
    slice_depth = 83 # 83px*2.41um/px = 200 um
    nSlicesx = math.ceil(len_x/slice_depth)
    nSlicesy = math.ceil(len_y/slice_depth)

    # create zarr arrays for each max projection
    sliced_maxx = slicedMaxProjectionsGroup.zeros('sliced_maxx',shape=(nSlicesx,len_t,len_z,len_y),chunks=(1,1,64,64))
    sliced_maxy = slicedMaxProjectionsGroup.zeros('sliced_maxy',shape=(nSlicesy,len_t,len_z,len_x),chunks=(1,1,64,64))

    for i in range(nSlicesx):
        for j in range(len_t):
            x_range = [i*slice_depth, (i+1)*slice_depth]
            frame = res_array[j, nChannel, :, :, x_range[0]:x_range[1]]
            sliced_maxx[i,j] = np.max(frame,axis=2)
    
    for i in range(nSlicesy):
        for j in range(len_t):
            y_range = [i*slice_depth, (i+1)*slice_depth]
            frame = res_array[j, nChannel, :, y_range[0]:y_range[1], :]
            sliced_maxy[i,j] = np.max(frame,axis=1)

def calcScaleMax(z):
    maxz = z['analysis']['max_projections']['maxz']
    scaleMax = np.max(maxz)
    return scaleMax

def makeOrthoMaxVideo(filename, z, scaleMax):
    
    maxz = z['analysis']['max_projections']['maxz']
    maxy = z['analysis']['max_projections']['maxy']
    maxx = z['analysis']['max_projections']['maxx']
    
    
    len_t = maxz.shape[0]
    len_z = maxy.shape[1]
    len_y = maxz.shape[1]
    len_x = maxz.shape[2]
    
    gap = 20

    xz = len_x + len_z + gap
    yz = len_y + len_z + gap

    imagingFreq = 10
    scaleBarY = yz - 76
    scaleBarX = len_x - 468
    scaleBarHeight = 30
    scaleBarLength = 416 # 1000um/2.41um/px = 416px
    scaleBarText = '1 mm'
    scaleBarText_offset = 50
    scaleBarZText = str(len_z * 2) + ' um'

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)
    #adjmax = 10000
    #adjust contrast based on max pixel value in the array
    adjmax = scaleMax
    adjmin = 0

    for i in tqdm(range(len_t)):

        # initialize frame 
        im = np.zeros([yz,xz])

        # copy max projections 
        im[0:len_z,0:len_x] = copy.copy(np.flip(maxy[i],axis=0))
        im[(len_z+gap):yz,0:len_x] = copy.copy(maxz[i])
        im[(len_z+gap):yz,(len_x+gap):xz] = copy.copy(np.transpose(maxx[i]))
        
        # adjust contrast
        im[np.where(im>adjmax)] = adjmax
        im[np.where(im<adjmin)] = adjmin
        backSub = im - adjmin
        backSub[np.where(backSub<0)] = 0
        scaledIm = np.divide(backSub,adjmax-adjmin)
        im8 = np.multiply(scaledIm,255).astype('uint8')
        frame = cv2.applyColorMap(im8,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]
        
        # time stamp
        t = f'{i*imagingFreq // 60:02d}' + ':' + f'{i*imagingFreq % 60:02d}'
        cv2.putText(frame,t,(25,len_z+gap+150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)

        # add xy scale bar 
        frame[scaleBarY:scaleBarY+scaleBarHeight,scaleBarX:scaleBarX+scaleBarLength,:] = 255
        cv2.putText(frame,scaleBarText,(scaleBarX+scaleBarText_offset,scaleBarY-30),cv2.FONT_HERSHEY_SIMPLEX,3,[255,255,255],6,cv2.LINE_AA)
        
        # add xz scale bar 
        frame[len_z-scaleBarHeight:len_z,len_x+gap:len_x+gap+len_z,:] = 255
        cv2.putText(frame,scaleBarZText,(len_x+gap,len_z-50),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,255],3,cv2.LINE_AA)
        
        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

def makeSlicedOrthoMaxVideos(filename, z, scaleMax):
        
    sliced_maxx = z['analysis']['sliced_max_projections']['sliced_maxx']

    nSlices = sliced_maxx.shape[0]
    len_t = sliced_maxx.shape[1]
    len_z = sliced_maxx.shape[2]
    len_y = sliced_maxx.shape[3]

    gap = 20

    size_x = len_y
    size_y = (len_z * nSlices) + (gap * (nSlices-1))

    imagingFreq = 10
    scaleBarY = size_y - 76
    scaleBarX = size_x - 468
    scaleBarHeight = 30
    scaleBarLength = 416 # 1000um/2.41um/px = 416px
    scaleBarText = '1 mm'
    scaleBarText_offset = 50

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(size_x,size_y),1)

    adjmax = scaleMax
    adjmin = 0

    for i in tqdm(range(len_t)):

        # initialize frame
        im = np.zeros([size_y,size_x])

        # copy max projections 
        for j in range(nSlices):
            im[(len_z*j+gap*j):(len_z*(j+1)+gap*j),:] = copy.copy(np.flip(sliced_maxx[j,i], axis=0))

        # adjust contrast
        im[np.where(im>adjmax)] = adjmax
        im[np.where(im<adjmin)] = adjmin
        backSub = im - adjmin
        backSub[np.where(backSub<0)] = 0
        scaledIm = np.divide(backSub,adjmax-adjmin)
        im8 = np.multiply(scaledIm,255).astype('uint8')
        frame = cv2.applyColorMap(im8,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]

        # time stamp
        t = f'{i*imagingFreq // 60:02d}' + ':' + f'{i*imagingFreq % 60:02d}'
        cv2.putText(frame,t,(25,150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)

        # add xy scale bar 
        frame[scaleBarY:scaleBarY+scaleBarHeight,scaleBarX:scaleBarX+scaleBarLength,:] = 255
        cv2.putText(frame,scaleBarText,(scaleBarX+scaleBarText_offset,scaleBarY-30),cv2.FONT_HERSHEY_SIMPLEX,3,[255,255,255],6,cv2.LINE_AA)

        # write frame
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

            



