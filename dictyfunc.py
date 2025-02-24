# Dicty data functions
# Edited by Jennifer Hill 11/26/2024 

import os, zarr, cv2, cmapy, copy
import numpy as np 
from skimage.filters import threshold_otsu, rank, threshold_mean, threshold_yen, threshold_triangle, threshold_isodata, threshold_li, threshold_minimum
from skimage.morphology import ball
from skimage.measure import label, regionprops
from tqdm import tqdm
# import matplotlib.pyplot as plt

def getChannelGroups(file,nChannel):
    
    groups = [string for string in list(file.group_keys()) if string.endswith(str(nChannel))]
    
    def sorting_key(string):
        _, _, value = string.partition('tp_')
        return int(value.split('_')[0])

    # Sort the strings based on the value after 'tp_'
    return sorted(groups, key=sorting_key)
    
def getAllGroups(file,nChannels):
    groups = []
    groups.append(getChannelGroups(file,i))
    return groups

def getDimensions(file,groups):
    frame = file[groups[0][0]]['s0']
    return len(groups[0]), frame.shape[0], frame.shape[1], frame.shape[2]

def getFrame(file,groups,channel,timepoint):
    return file[groups[channel][timepoint]]['s0']

def convertRawData(file,z,nChannels,groups):
    
    # get dataset dimensions 
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # define array 
    rawData = root.zeros('raw_data',
                         shape=(nChannels,len_t,len_z,len_y,len_x),
                         chunks = (1,1,64,64,64),
                         dtype='uint64')
    
    # fill array 
    for j in range(nChannels):
        for i in range(len_t):
            rawData[j,i] = z[groups[j][i]]['s0']

def calcMaxProjections(file,z,nChannel,groups):
    
    # get dataset dimensions 
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # create analysis group if it does not already exist 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    maxProjectionsGroup = analysis_group.create_group('max_projections')
    
    # create zarr arrays for each max projection 
    maxz = maxProjectionsGroup.zeros('maxz',shape=(len_t,len_y,len_x),chunks=(1,64,64))
    maxx = maxProjectionsGroup.zeros('maxx',shape=(len_t,len_z,len_y),chunks=(1,64,64))
    maxy = maxProjectionsGroup.zeros('maxy',shape=(len_t,len_z,len_x),chunks=(1,64,64))
    
    # iterate through each timepoint and compute max projections
    for i in tqdm(range(len_t)):
        frame = z[groups[nChannel][i]]['s0']
        maxz[i] = np.max(frame,axis=0)
        maxx[i] = np.max(frame,axis=2)
        maxy[i] = np.max(frame,axis=1)

# added by JH to make movies colored by z-depth 
def calcMaxProjectionsZDepth(file,z,nChannel,groups):
    # get dataset dimensions 
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # create analysis group if it does not already exist 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    maxProjectionsZGroup = analysis_group.create_group('max_projections_z')
    
    # create zarr arrays for each max projection 
    maxz = maxProjectionsZGroup.zeros('maxz',shape=(len_t,len_y,len_x,2),chunks=(1,64,64,2))
    maxx = maxProjectionsZGroup.zeros('maxx',shape=(len_t,len_z,len_y,2),chunks=(1,64,64))
    maxy = maxProjectionsZGroup.zeros('maxy',shape=(len_t,len_z,len_x,2),chunks=(1,64,64))
    
    # iterate through each timepoint and compute max projections and z-index for maxz projection
    for i in tqdm(range(len_t)):
        frame = z[groups[nChannel][i]]['s0']
        maxz[i] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
        maxx[i] = np.max(frame,axis=2)
        maxy[i] = np.max(frame,axis=1)

def threshOtsu(file,z,nChannels,groups):
    
    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    otsu_cells = analysis_group.zeros(
        'otsu_global_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i))
                
        # threshold 
        thresh = threshold_otsu(frame)
                
        # add to array 
        otsu_cells[i] = frame > thresh
    
    # create zarr array for rocks channel 
    otsu_rocks = analysis_group.zeros(
        'otsu_global_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,1,i))
                
        # threshold 
        thresh = threshold_otsu(frame)
                
        # add to array 
        otsu_rocks[i] = frame < thresh

def threshOtsuLocal(file,z,nChannels,groups,radius):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    otsu_cells = analysis_group.zeros(
        'otsu_local_cells_r%.3i' % radius,
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        otsu_cells[i] = rank.otsu(frame,footprint)   

    # create zarr array for rocks channel 
    otsu_rocks = analysis_group.zeros(
        'otsu_local_rocks_r%.3i' % radius,
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        otsu_rocks[i] = ~rank.otsu(frame,footprint)
        
def threshOtsuDiffDir(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    otsu_cells = analysis_group.zeros(
        'otsu_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        #otsu_cells[i] = rank.otsu(frame,footprint)   
        otsu_thresh = threshold_otsu(frame)
        otsu_cells[i] = frame > otsu_thresh

    # create zarr array for rock channel 
    otsu_rocks = analysis_group.zeros(
        'otsu_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        #otsu_rocks[i] = ~rank.otsu(frame,footprint)
        otsu_thresh = threshold_otsu(frame)
        otsu_rocks[i] = frame < otsu_thresh
        
def threshTest(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    cells = analysis_group.zeros(
        'test',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
        
def threshMean(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    mean_cells = analysis_group.zeros(
        'mean_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        # mean_cells[i] = rank.mean(frame,footprint)   
        mean_thresh = threshold_mean(frame)
        mean_cells[i] = frame > mean_thresh

    # create zarr array for rock channel 
    mean_rocks = analysis_group.zeros(
        'mean_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        # mean_rocks[i] = ~rank.mean(frame,footprint)
        mean_thresh = threshold_mean(frame)
        mean_rocks[i] = frame < mean_thresh
        
def threshYen(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    yen_cells = analysis_group.zeros(
        'yen_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        #yen_cells[i] = rank.yen(frame,footprint)
        yen_thresh = threshold_yen(frame)
        yen_cells[i] = frame > yen_thresh

    # create zarr array for rock channel 
    yen_rocks = analysis_group.zeros(
        'yen_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        #yen_rocks[i] = ~rank.yen(frame,footprint)
        yen_thresh = threshold_yen(frame)
        yen_rocks[i] = frame < yen_thresh
        
def threshTriangle(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    triangle_cells = analysis_group.zeros(
        'triangle_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        #triangle_cells[i] = rank.triangle(frame,footprint)
        triangle_thresh = threshold_triangle(frame)
        triangle_cells[i] = frame > triangle_thresh

    # create zarr array for rock channel 
    triangle_rocks = analysis_group.zeros(
        'triangle_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        #triangle_rocks[i] = ~rank.triangle(frame,footprint)
        triangle_thresh = threshold_triangle(frame)
        triangle_rocks[i] = frame < triangle_thresh
        
def threshIsodata(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    isodata_cells = analysis_group.zeros(
        'isodata_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        #isodata_cells[i] = rank.isodata(frame,footprint)
        isodata_thresh = threshold_isodata(frame)
        isodata_cells[i] = frame > isodata_thresh

    # create zarr array for rock channel 
    isodata_rocks = analysis_group.zeros(
        'isodata_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        #isodata_rocks[i] = ~rank.isodata(frame,footprint)
        isodata_thresh = threshold_isodata(frame)
        isodata_rocks[i] = frame < isodata_thresh
        
def threshLi(file,z,nChannels,groups):

    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    # see if analysis group exists 
    if 'analysis' in z:
        analysis_group = root['analysis']
    else:
        analysis_group = root.create_group('analysis')
    
    # create zarr array for cell channel 
    li_cells = analysis_group.zeros(
        'li_local_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # define footprint to use for local thresholding 
    #footprint = ball(radius)
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get cell frame 
        frame = np.asarray(getFrame(z,groups,0,i),dtype='uint16')
                
        # apply local threshold and add to array
        #li_cells[i] = rank.li(frame,footprint)
        li_thresh = threshold_li(frame)
        li_cells[i] = frame > li_thresh

    # create zarr array for rock channel 
    li_rocks = analysis_group.zeros(
        'li_local_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='bool')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        # get rock frame 
        frame = np.asarray(getFrame(z,groups,1,i),dtype='uint16')
                
        # threshold, invert, and add to array  
        #li_rocks[i] = ~rank.li(frame,footprint)
        li_thresh = threshold_li(frame)
        li_rocks[i] = frame < li_thresh

def labelImage(file,z,nChannels,groups):
    
    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # create nested directory structure within analysis group
    nested_store = zarr.NestedDirectoryStore(file,dimension_separator='/')
    root = zarr.group(store=nested_store,overwrite=False)
    
    analysis_group = root['analysis']
    
    # add label array to analysis group 
    cells_array = analysis_group.zeros(
        'label_cells',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='uint16')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        frame = analysis_group['otsu_global_cells'][i]
        cells_array[i] = label(frame)
    
    # add rocks label array to analysis group 
    rocks_array = analysis_group.zeros(
        'label_rocks',
        shape=(len_t,len_z,len_y,len_x),
        chunks=(1,64,64,64),
        dtype='uint16')
    
    # iterate through timepoints 
    for i in tqdm(range(len_t)):
        frame = analysis_group['otsu_global_rocks'][i]
        rocks_array[i] = label(frame)

def cellClumpSizeOverTime(z,groups):
    
    # create groups
    analysis_group = z['analysis']
    cellClumpAreaGroup = analysis_group.create_group('cellRegionArea')
    # cellClumpAreaConvexGroup = analysis_group.create_group('cellRegionAreaConvex')
    # cellClumpAreaFilledGroup = analysis_group.create_group('cellRegionAreaFilled') 
    
    # get dimensions of dataset
    len_t,len_z,len_y,len_x = getDimensions(z,groups)
    
    # iterate through each time point 
    for i in tqdm(range(len_t)):
        
        # get region props for each cell clump at time point 
        regions = regionprops(z['analysis']['label_cells'][i])
        
        # initialize arrays to fill with values 
        areas = np.zeros(len(regions))
        # areas_convex = np.zeros_like(areas)
        # areas_filled = np.zeros_like(areas)
        
        # fill arrays with values 
        for j in range(len(regions)):
            areas[j] = regions[j].area
            # areas_convex[j] = regions[j].area_convex
            # areas_filled[j] = regions[j].area_filled
        
        # add arrays to each group 
        cellClumpAreaGroup.array('t%.3i' % i,data=areas)
        # cellClumpAreaConvexGroup.array('t%.3i' % i,data=areas_convex)
        # cellClumpAreaFilledGroup.array('t%.3i' % i,data=areas_filled)
        
        

def threshCheck(exp_dir,file,groups,nT,nZ):
    
    # create image folder if not created already 
    img_dir = exp_dir + '/seg_images'
    if not os.path.exists(img_dir):
        os.chdir(exp_dir)
        os.makedirs(img_dir)
    os.chdir(img_dir)
    
    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(file,groups)
    
    for i in tqdm(np.linspace(0,len_t-1,nT).astype('int')):
        cells_raw = getFrame(file,groups,0,i)
        rocks_raw = getFrame(file,groups,1,i)
        
        for j in np.linspace(0,len_z-1,nZ).astype('int'):
            slice_cells_raw = cells_raw[j]
            slice_cells_seg = file['binary_seg']['binary_seg'][i,0,j]
            slice_rocks_raw = rocks_raw[j]
            slice_rocks_seg = file['binary_seg']['binary_seg'][i,1,j]
            
            plt.figure(figsize=(14,9))
            plt.subplot(2,3,1)
            plt.title('cells_raw')
            plt.imshow(slice_cells_raw,vmin=0,vmax=10000,cmap='viridis')
            
            plt.subplot(2,3,2)
            plt.title('cells_seg')
            plt.imshow(slice_cells_seg,cmap='Reds',interpolation='none')
            
            plt.subplot(2,3,3)
            plt.title('cells_overlay')
            plt.imshow(slice_cells_raw,vmin=0,vmax=10000,cmap='viridis')
            plt.imshow(slice_cells_seg,cmap='Reds',alpha=0.5,interpolation='none')
            
            plt.subplot(2,3,4)
            plt.title('rocks_raw')
            plt.imshow(slice_rocks_raw,vmin=0,vmax=30000,cmap='gray')
            
            plt.subplot(2,3,5)
            plt.title('rocks_seg')
            plt.imshow(slice_rocks_seg,cmap='Reds')
            
            plt.subplot(2,3,6)
            plt.title('rocks_overlay')
            plt.imshow(slice_rocks_raw,vmin=0,vmax=30000,cmap='gray')
            plt.imshow(slice_rocks_seg,cmap='Reds',alpha=0.5)
            
            plt.tight_layout()
            
            plt.savefig('t%i_z%i.png' % (i,j))
                

def makeOrthoMaxVideo(filename,z):
    
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
    scaleBarLength = 416
    scaleBarText = '1 mm'
    scaleBarText_offset = 50
    scaleBarZText = str(len_z * 2) + ' um'

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)
    #adjmax = 10000
    #adjust contrast based on max pixel value in the array
    adjmax = np.max(maxz)
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
        cv2.putText(frame,scaleBarZText,(len_x+gap+scaleBarText_offset,len_z-50),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],3,cv2.LINE_AA)
        
        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

def saveFrame(exp_dir,file,t,c):
    
    # create image folder if not created already 
    img_dir = exp_dir + '/3d_views'
    if not os.path.exists(img_dir):
        os.chdir(exp_dir)
        os.makedirs(img_dir)
    os.chdir(img_dir)
    
    slice_cells_seg = file['binary_seg']['binary_seg'][t,c]
    ax = plt.figure(figsize=(8,5)).add_subplot(projection='3d')
    ax.voxels(slice_cells_seg)
    ax.savefig('cell_seg_c%i_t%i.png' % (c,t))

def saveLabelFrame(exp_dir,file,groups,t,c):
    
    # create image folder if not created already 
    img_dir = exp_dir + '/3d_views'
    if not os.path.exists(img_dir):
        os.chdir(exp_dir)
        os.makedirs(img_dir)
    os.chdir(img_dir)
    
    # get data dimensions
    len_t,len_z,len_y,len_x = getDimensions(file,groups)
    maxdim = np.max((len_z,len_y,len_x))
    
    # get slice of label cell array 
    frame = np.asarray(file['binary_seg']['label'][t,c])
    frame = np.transpose(frame, (2, 1, 0))
    nObjects = np.max(frame)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    for i in range(1,nObjects):
        clump = np.zeros_like(frame,dtype='bool')
        clump[frame==i] = True
        ax.voxels(clump,shade=1,facecolor=plt.cm.viridis(np.random.randint(nObjects)/nObjects))
    ax.set_xlim([0,maxdim])
    ax.set_ylim([0,maxdim])
    ax.set_zlim([0,maxdim])
    ax.view_init(elev=1.25*len_z, azim=-30, roll=0)
    ax.axis('off')
    
    plt.savefig('cell_label_c%i_t%i.png' % (c,t))

def makeTopBottomOrthoMaxVideo(filename,maxz,maxy,maxx,len_t,len_z,len_y,len_x):
    
    zcut = len_z // 2

    maxx_bottom = np.zeros([len_t,zcut,len_y])
    maxy_bottom = np.zeros([len_t,zcut,len_x])
    maxz_bottom = np.zeros([len_t,len_y,len_x])

    maxx_top = np.zeros([len_t,len_z-zcut,len_y])
    maxy_top = np.zeros([len_t,len_z-zcut,len_x])
    maxz_top = np.zeros([len_t,len_y,len_x])

    for i in range(len(c0_groups)):
        frame = z[groups[i]]['s0']
        
        bottom = frame[0:zcut]
        top = frame[zcut::]
        
        maxx_bottom[i] = np.max(bottom,axis=2)
        maxy_bottom[i] = np.max(bottom,axis=1)
        maxz_bottom[i] = np.max(bottom,axis=0)
        
        maxx_top[i] = np.max(top,axis=2)
        maxy_top[i] = np.max(top,axis=1)
        maxz_top[i] = np.max(top,axis=0) 

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)
    adjmax = 10000
    adjmin = 0

    gap = 20
    gap2 = 100

    len_x = ref_frame.shape[2]
    len_y = ref_frame.shape[1]
    len_z = np.maximum(zcut,totalz-zcut)

    for i in tqdm(range(len(maxz))):

        # initialize frame 
        im = np.zeros([len_z + gap + len_y,2*(len_x+gap+len_z)+gap2])

        # bottom 
        im[totalz-2*zcut:len_z,0:len_x] = copy.copy(np.flip(maxy_bottom[i],axis=0))
        im[(len_z+gap)::,0:len_x] = copy.copy(maxz_bottom[i])
        im[(len_z+gap)::,(len_x+gap):(len_x+gap+zcut)] = copy.copy(np.transpose(maxx_bottom[i]))

        # top 
        im[0:len_z,len_x+gap+len_z+gap2:len_x+gap+len_z+gap2+len_x] = copy.copy(np.flip(maxy_top[i],axis=0))
        im[(len_z+gap)::,len_x+gap+len_z+gap2:len_x+gap+len_z+gap2+len_x] = copy.copy(maxz_top[i])
        im[(len_z+gap)::,len_x+gap+len_z+gap2+len_x+gap::] = copy.copy(np.transpose(maxx_top[i]))
        
        # adjust contrast
        im[np.where(im>adjmax)] = adjmax
        im[np.where(im<adjmin)] = adjmin
        backSub = im - adjmin
        backSub[np.where(backSub<0)] = 0
        scaledIm = np.divide(backSub,adjmax-adjmin)
        im8 = np.multiply(scaledIm,255).astype('uint8')
        frame = cv2.applyColorMap(im8,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]
        
        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()