# Dicty data functions for ome-zarr datasets

import copy
import os
import traceback

import xml.etree.ElementTree as et
import zarr
import cv2
import cmapy
import json
from PIL import ImageFont, ImageDraw, Image
from dask.distributed import Client, wait
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def createRootStore(zarrFile):
    nestedStore = zarr.NestedDirectoryStore(zarrFile, dimension_separator='/')
    root = zarr.group(store=nestedStore, overwrite=False)

def createZarrGroup(root, groupName):
    if groupName in root:
        group = root[groupName]
    else:
        group = root.create_group(groupName)
    return group

class channel:
    def __init__(self, name, nChannel, scaleMax, scaleMin, gamma=1.0, invertChannel=False, voxelDims=None):
        self.name = name
        self.nChannel = nChannel
        self.scaleMax = scaleMax
        self.scaleMin = scaleMin
        self.gamma = gamma
        self.invertChannel = invertChannel
        self.voxelDims = voxelDims

def getChannelsFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    channels = []
    for channelInfo in channelSpecs:
        currentChannel = channel(name=channelInfo["name"],
                                nChannel=channelInfo["channelNumber"],
                                scaleMax=channelInfo["scaleMax"],
                                scaleMin=channelInfo["scaleMin"],)
        if "gamma" in channelInfo:
            currentChannel.gamma = channelInfo["gamma"]
        if "invertChannel" in channelInfo:
            currentChannel.invertChannel = channelInfo["invertChannel"]
        channels.append(currentChannel)
    return channels

def getCroppingDimsFromJSON(jsonFile):
    with open(jsonFile) as f:
        try:
            croppingParams = json.load(f)["croppingParameters"]
            if croppingParams:
                cropX = croppingParams.get("cropX", [None, None])
                cropY = croppingParams.get("cropY", [None, None])
                cropZ = croppingParams.get("cropZ", [None, None])
            return [cropX, cropY, cropZ]
        except Exception as e:
            return None


def getSliceDepthFromJSON(jsonFile):
    with open(jsonFile) as f:
        movieSpecs = json.load(f)["movieSpecs"]
        if "sliceDepth" in movieSpecs:
            sliceDepth = movieSpecs["sliceDepth"]
        else:
            sliceDepth = "auto"
    return sliceDepth

def getImagingFreqFromJSON(jsonFile):
    with open(jsonFile) as f:
        imagingFreq = json.load(f)["imagingParameters"]["imagingFrequency"]
    return imagingFreq

def getVoxelDimsFromXML(xmlFile, res_lvl=0):
    XMLTree = et.parse(xmlFile)
    XMLRoot = XMLTree.getroot()
    imageMetaData = XMLRoot[res_lvl][0]
    pixelSizeX = float(imageMetaData.get('PhysicalSizeX'))
    pixelSizeY = float(imageMetaData.get('PhysicalSizeY'))
    pixelSizeZ = float(imageMetaData.get('PhysicalSizeZ'))
    return [pixelSizeX, pixelSizeY, pixelSizeZ]


def calcMaxProjections(root, maxProjectionsRoot, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine resArray
    cropDims = getCroppingDimsFromJSON(maxProjectionsRoot.store.path + '/../../parameters.json')
    if cropDims:
        resArray = resArray[:, :, cropDims[2][0]:cropDims[2][1], cropDims[1][0]:cropDims[1][1], cropDims[0][0]:cropDims[0][1]]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    # create zarr arrays for each max projection 
    maxZ = maxProjectionsRoot.zeros('maxz',shape=(lenT,lenCh,2,lenY,lenX),chunks=(1,lenCh,2,lenY,lenX))
    maxX = maxProjectionsRoot.zeros('maxx',shape=(lenT,lenCh,lenZ,lenY),chunks=(1,lenCh,lenZ,lenY))
    maxY = maxProjectionsRoot.zeros('maxy',shape=(lenT,lenCh,lenZ,lenX),chunks=(1,lenCh,lenZ,lenX))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            frame = resArray[i, j, :, :, :]
            maxZ[i,j] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
            maxX[i,j] = np.max(frame,axis=2)
            maxY[i,j] = np.max(frame,axis=1)  

def calcOpticalFlowMaxProjections(maxProjectionsRoot, opticalFolder, cropID=''):
    """Create max projections specifically from optical flow RGBM (rgb and magnitude) arrays"""
    
    print(f"🔍 Looking for optical flow data in: {opticalFolder}")
    
    # Find all frame directories
    frame_dirs = []
    if os.path.exists(opticalFolder):
        for item in os.listdir(opticalFolder):
            item_path = os.path.join(opticalFolder, item)
            if os.path.isdir(item_path) and item.isdigit():
                frame_dirs.append(int(item))
    
    if not frame_dirs:
        print(f"No frame directories found in {opticalFolder}")
        return
    
    frame_dirs.sort()
    print(f"📊 Found {len(frame_dirs)} frames: {frame_dirs[:5]}{'...' if len(frame_dirs) > 5 else ''}")
    
    # Get dimensions from first available RGBM file
    sample_frame = frame_dirs[0]
    sample_rgbm_file = os.path.join(opticalFolder, str(sample_frame), "optical_flow_rgbm.npy")
    
    if not os.path.exists(sample_rgbm_file):
        print(f"Sample RGBM file not found: {sample_rgbm_file}")
        return
    
    # Load sample to get dimensions
    sample_rgbm = np.load(sample_rgbm_file)
    lenZ, lenY, lenX, num_channels = sample_rgbm.shape
    lenT = len(frame_dirs)
    
    print(f"📐 Dataset dimensions:")
    print(f"   Frames (T): {lenT}")
    print(f"   Z-slices: {lenZ}")
    print(f"   Y: {lenY}")
    print(f"   X: {lenX}")
    print(f"   Channels: {num_channels} (RGBM)")
    
    # Check for cropping parameters
    parentDir = os.path.dirname(opticalFolder)
    cropDims = getCroppingDimsFromJSON(os.path.join(parentDir, 'parameters.json'))
    if cropDims:
        print(f"🔧 Applying cropping: {cropDims}")
        lenZ = cropDims[2][1] - cropDims[2][0]
        lenY = cropDims[1][1] - cropDims[1][0]
        lenX = cropDims[0][1] - cropDims[0][0]
        print(f"📐 Cropped dimensions: Z={lenZ}, Y={lenY}, X={lenX}")
    
    # Create zarr arrays for optical flow RGBM max projections
    print("📝 Creating zarr arrays for max projections...")
    
    flowMaxZ = maxProjectionsRoot.zeros('flow_maxz', shape=(lenT, 4, lenY, lenX), 
                                       chunks=(1, 4, lenY, lenX))
    flowMaxX = maxProjectionsRoot.zeros('flow_maxx', shape=(lenT, 4, lenZ, lenY), 
                                       chunks=(1, 4, lenZ, lenY))
    flowMaxY = maxProjectionsRoot.zeros('flow_maxy', shape=(lenT, 4, lenZ, lenX), 
                                       chunks=(1, 4, lenZ, lenX))
    
    print("Zarr arrays created successfully")
    
    # Iterate through each timepoint and compute max projections
    print("🎨 Computing max projections...")
    
    for i, frame_num in enumerate(tqdm(frame_dirs, desc="Processing frames")):
        frame_dir = os.path.join(opticalFolder, str(frame_num))
        rgbm_file = os.path.join(frame_dir, "optical_flow_rgbm.npy")
        
        if os.path.exists(rgbm_file):
            try:
                # Load RGBM data: (Z, Y, X, 4) where 4 = [Red, Green, Blue, Magnitude]
                rgbm_data = np.load(rgbm_file)
                
                # Apply cropping if specified
                if cropDims:
                    rgbm_data = rgbm_data[cropDims[2][0]:cropDims[2][1], 
                                         cropDims[1][0]:cropDims[1][1], 
                                         cropDims[0][0]:cropDims[0][1], :]
                
                # Compute max projections for each channel (R, G, B, M)
                for j in range(4):
                    channel_data = rgbm_data[:, :, :, j]  # (Z, Y, X)
                    
                    # Max projections along each axis
                    flowMaxZ[i, j] = np.max(channel_data, axis=0)  # (Y, X) - along Z
                    flowMaxX[i, j] = np.max(channel_data, axis=2)  # (Z, Y) - along X
                    flowMaxY[i, j] = np.max(channel_data, axis=1)  # (Z, X) - along Y
                    
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                # Fill with zeros for this frame
                flowMaxZ[i, :] = 0
                flowMaxX[i, :] = 0
                flowMaxY[i, :] = 0
        else:
            print(f"RGBM file not found for frame {frame_num}")
            # Fill with zeros for this frame
            flowMaxZ[i, :] = 0
            flowMaxX[i, :] = 0
            flowMaxY[i, :] = 0
    
    print("Optical flow max projections completed!")
    print(f"Created arrays:")
    print(f"   flow_maxz: {flowMaxZ.shape}")
    print(f"   flow_maxx: {flowMaxX.shape}")
    print(f"   flow_maxy: {flowMaxY.shape}")

def calcMaxs(resArray, i, j, maxZ, maxX, maxY):
    frame = resArray[i, j, :, :, :]
    maxZ[i,j] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
    maxX[i,j] = np.max(frame,axis=2)
    maxY[i,j] = np.max(frame,axis=1)    

def calcMaxProjectionsDask(root, maxProjectionsRoot, client, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine resArray
    cropDims = getCroppingDimsFromJSON(maxProjectionsRoot.store.path + '/../../parameters.json')
    if cropDims:
        resArray = resArray[:, :, cropDims[2][0]:cropDims[2][1], cropDims[1][0]:cropDims[1][1], cropDims[0][0]:cropDims[0][1]]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    # create zarr arrays for each max projection 
    maxZ = maxProjectionsRoot.zeros('maxz',shape=(lenT,lenCh,2,lenY,lenX),chunks=(1,lenCh,2,lenY,lenX))
    maxX = maxProjectionsRoot.zeros('maxx',shape=(lenT,lenCh,lenZ,lenY),chunks=(1,lenCh,lenZ,lenY))
    maxY = maxProjectionsRoot.zeros('maxy',shape=(lenT,lenCh,lenZ,lenX),chunks=(1,lenCh,lenZ,lenX))

    # iterate through each timepoint and compute max projections
    futures = []
    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            futures.append(client.submit(calcMaxs, resArray, i, j, maxZ, maxX, maxY))
    wait(futures)    


def calcSlicedMaxProjections(root, slicedMaxProjectionsRoot, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine resArray
    cropDims = getCroppingDimsFromJSON(slicedMaxProjectionsRoot.store.path + '/../../parameters.json')
    if cropDims:
        resArray = resArray[:, :, cropDims[2][0]:cropDims[2][1], cropDims[1][0]:cropDims[1][1], cropDims[0][0]:cropDims[0][1]]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    # get slice depth from parameters.json
    sliceDepth = getSliceDepthFromJSON(slicedMaxProjectionsRoot.store.path + '/../../parameters.json')
    voxelDims = getVoxelDimsFromXML(root.store.path + '/OME/METADATA.ome.xml')

    # set number of slices
    if sliceDepth == "auto":
        nSlicesX = 20
        nSlicesY = 20
        sliceDepthX = lenX//(nSlicesX-1)
        sliceDepthY = lenY//(nSlicesY-1)
    else:
        sliceDepthX = int(sliceDepth//voxelDims[0])
        nSlicesX = lenX//sliceDepthX
        sliceDepthY = int(sliceDepth//voxelDims[1])
        nSlicesY = lenY//sliceDepthY

    # create zarr arrays for each max projection
    slicedMaxX = slicedMaxProjectionsRoot.zeros('sliced_maxx',shape=(lenT,lenCh,nSlicesX,lenZ,lenY),chunks=(1,1,2,lenZ,lenY))
    slicedMaxY = slicedMaxProjectionsRoot.zeros('sliced_maxy',shape=(lenT,lenCh,nSlicesY,lenZ,lenX),chunks=(1,1,2,lenZ,lenX))

    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            for k in range(nSlicesX-1):
                rangeX = [k*sliceDepthX, (k+1)*sliceDepthX]
                frame = resArray[i,j,:,:,rangeX[0]:rangeX[1]]
                slicedMaxX[i,j,k] = np.max(frame,axis=2)
            #fill last chunk with the rest of the data
            rangeX = [(nSlicesX-1)*sliceDepthX, lenX]
            frame = resArray[i,j,:,:,rangeX[0]:rangeX[1]]
            slicedMaxX[i,j,nSlicesX-1] = np.max(frame,axis=2)

            for k in range(nSlicesY-1):
                rangeY = [k*sliceDepthY, (k+1)*sliceDepthY]
                frame = resArray[i,j,:,rangeY[0]:rangeY[1],:]
                slicedMaxY[i,j,k] = np.max(frame,axis=1)
            #fill last chunk with the rest of the data
            rangeY = [(nSlicesY-1)*sliceDepthY, lenY]
            frame = resArray[i,j,:,rangeY[0]:rangeY[1],:]
            slicedMaxY[i,j,nSlicesY-1] = np.max(frame,axis=1)

def generateUniqueFilename(filename, ext):
    i = 1
    while os.path.exists(filename + ext):
        if i == 1:
            filename = filename +'_1'
        else:
            filename = filename[:-1] + str(i)
        i += 1
    return filename + ext

def calcAutoContrast(maxProj, nChannel):
    firstFrame = maxProj[0, nChannel, 0]

    # define histogram of pixel values in first frame
    histMin = np.min(firstFrame)
    histMax = np.max(firstFrame)
    binSize = (histMax - histMin) / 256
    histogram = np.histogram(firstFrame, bins = 256, range=(histMin, histMax))[0]

    # define limit and threshold
    height, width = firstFrame.shape[:2]
    pixelCount = height * width
    limit = pixelCount/10
    threshold = int(pixelCount/5000)

    # find the bin for the min and max contrast values
    bin = -1
    foundMinBin = False
    while not foundMinBin and bin < 255:
        bin += 1
        countInBin = histogram[bin]
        if countInBin > limit:
            countInBin = 0
        foundMinBin = countInBin > threshold
    hMinBin = bin
    
    bin = 256
    foundMaxBin = False
    while not foundMaxBin and bin > 0:
        bin -= 1
        countInBin = histogram[bin]
        if countInBin > limit:
            countInBin = 0
        foundMaxBin = countInBin > threshold
    hMaxBin = bin

    # find scaleMin and scaleMax based on hMinBin and hMaxBin
    if hMaxBin > hMinBin:
        scaleMin = histMin + (hMinBin * binSize)
        scaleMax = histMin + (hMaxBin * binSize)
    # bad cases: hMaxBin is same or less than hMinBin, just use the min and max of the histogram
    else:
        scaleMin = histMin
        scaleMax = histMax

    print('scaleMin:', scaleMin)
    print('scaleMax:', scaleMax)

    return scaleMin, scaleMax
    

def getProjectionDimensions(root):
    # return the dimensions of the max projections
    if 'maxx' in root:
        maxX = root['maxx']
        maxY = root['maxy']
    else:
        maxX = root['sliced_maxx']
        maxY = root['sliced_maxy']
    lenT = maxX.shape[0]
    lenZ = maxX.shape[-2]
    lenY = maxX.shape[-1]
    lenX = maxY.shape[-1]
    return lenT, lenZ, lenY, lenX

def adjustContrast(im, scaleMax, scaleMin, gamma):
    im = np.clip(im,scaleMin,scaleMax)
    backSub = im - scaleMin
    backSub[np.where(backSub<0)] = 0
    scaledIm = np.divide(backSub,scaleMax-scaleMin)
    gammaCorrectedIm = np.power(scaledIm,gamma)
    contrastedIm = np.multiply(gammaCorrectedIm,255).astype('uint8')
    return(contrastedIm)


class scaleBar:
    def __init__(self, posY, posX, length, pxPerMicron, font):
        self.posY = posY
        self.posX = posX
        self.length = length
        self.pxPerMicron = pxPerMicron
        self.font = font
        self.lengthInPx = round(length*pxPerMicron)
        self.heightInPx = int((length*pxPerMicron)//10)
        self.units = '\u03BCm'
        self.text = str(self.length) + self.units

    def _setFont(self):
        #TODO: add a style sheet and get font from there
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        fontPath = os.path.join(moduleDir, '..', 'fonts', 'Lato2OFL', 'Lato-Black.ttf')
        fontSize = 1
        self.font = ImageFont.truetype(fontPath, fontSize)
        while self.font.getlength(self.text) <= self.lengthInPx/2:
            fontSize += 1
            self.font = ImageFont.truetype(fontPath, fontSize)
        return

    def _addScaleBar(self, frame):
        # add bar
        frame[self.posY-self.heightInPx:self.posY, self.posX-self.lengthInPx:self.posX,:] = 255
        # convert length to mm if larger than 1000
        if self.length >= 1000:
            self.text = str(int(self.length/1000)) + 'mm'
        # add text
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        textWidth = draw.textlength(self.text, font=self.font)
        textHeight, _ = self.font.getmetrics()
        textPos = (self.posX-(self.lengthInPx//2)-(textWidth//2), self.posY-round(self.heightInPx*1.3)-textHeight)
        draw.text(textPos, self.text, font=self.font, fill=(255, 255, 255))
        frame = np.array(img_pil)
        return frame
      
def getScaleBarLength(root, voxelDims):
    #TODO: add scaling factor for sliced movies where the scale bar should be smaller
    #approxScaleBarLength = projDimsUM[1]/scaleFactor
    #alternatively, for sliced movies, projDims should be switched out with movieDims
    scaleBarLengths = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000] # in um

    projDimsPx = getProjectionDimensions(root)
    projDimsUm = [projDimsPx[3]*voxelDims[0], projDimsPx[2]*voxelDims[1], projDimsPx[1]*voxelDims[2]]
    approxScaleBarLength = projDimsUm[1]/5
    scaleBarLength = min(scaleBarLengths, key=lambda x:abs(x-approxScaleBarLength))
    return scaleBarLength


def calculateTimeStamp(frame_index, imaging_freq, total_frames):
    """
    Calculate timestamp string from frame index and imaging frequency.
    
    Args:
        frame_index (int): Frame number (0-based)
        imaging_freq (float): Time between frames in minutes
        
    Returns:
        str: Formatted timestamp string
    """
    # Calculate maximum timestamp to determine format
    max_total_minutes = (total_frames - 1) * imaging_freq
    max_hours = int(max_total_minutes // 60)
    max_remaining_minutes = max_total_minutes % 60
    max_minutes = int(max_remaining_minutes)
    max_seconds = int((max_remaining_minutes - max_minutes) * 60)

    show_hours = max_hours > 0
    show_seconds = max_seconds > 0

    # Calculate current timestamp
    total_minutes = frame_index * imaging_freq
    hours = int(total_minutes // 60)
    remaining_minutes = total_minutes % 60
    minutes = int(remaining_minutes)
    seconds = int((remaining_minutes - minutes) * 60)
    
    # Format based on maximum needed format
    if show_hours:
        if show_seconds:
            return f'{hours:02d}h:{minutes:02d}m:{seconds:02d}s'
        else:
            return f'{hours:02d}h:{minutes:02d}m'
    else:
        if show_seconds:
            return f'{minutes:02d}m:{seconds:02d}s'
        else:
            return f'{minutes:02d}m'


def addTimeStamp(frame, timeStampPos, t, font):
    # time stamp
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(timeStampPos, t, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)
    return frame

def scaleXZYZ(im, zToXYRatio):
    # scale the XZ and YZ projections to match the XY projection
    scaledIm = np.zeros([int(round(im.shape[0]*zToXYRatio)), im.shape[1]])
    for i in range(im.shape[0]):
        scaledIm[int(round(i*zToXYRatio)):int(round((i+1)*zToXYRatio)),:] = im[i,:]
    return scaledIm


def makeOrthoMaxVideoClean(root, channel, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + '_orthomax_clean_' + cmap, ext)
    nChannel = channel.nChannel
    voxelDims = channel.voxelDims
    scaleMax = channel.scaleMax
    scaleMin = channel.scaleMin
    gamma = channel.gamma

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')

    maxZ = root['maxz']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)

    movieWidth = lenX 
    movieHeight = lenY

    # calc scaleMin, scaleMax, and gamma if not provided
    if scaleMin == "None" or scaleMax == "None":
        scaleMin, scaleMax = calcAutoContrast(maxZ, nChannel)
    if gamma == "None":
        gamma = 1
    
    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try: 
        for i in tqdm(range(lenT)):

            # initialize frame 
            im = np.zeros([movieHeight,movieWidth])

            # copy max projections 
            im[:movieHeight,0:lenX] = copy.copy(maxZ[i,nChannel,0])
            
            contrastedIm = adjustContrast(im, scaleMax, scaleMin, gamma)

            # invert if rock channel
            if channel.name == 'rocks':
                contrastedIm = 255 - contrastedIm

            frame = cv2.applyColorMap(contrastedIm,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def makeOrthoMaxVideo(root, channel, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + '_orthomax_' + cmap, ext)
    nChannel = channel.nChannel
    voxelDims = channel.voxelDims
    scaleMax = channel.scaleMax
    scaleMin = channel.scaleMin
    gamma = channel.gamma

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    maxZ = root['maxz']
    maxY = root['maxy']
    maxX = root['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)

    # calc scaled Z dimension
    zToXYRatio = voxelDims[2]/voxelDims[0]
    scaledLenZ = int(round(lenZ*zToXYRatio))
    
    gap = 20

    movieWidth = lenX + scaledLenZ + gap
    movieHeight = lenY + scaledLenZ + gap

    # calc scaleMin, scaleMax, and gamma if not provided
    if scaleMin == "None" or scaleMax == "None":
        scaleMin, scaleMax = calcAutoContrast(maxZ, nChannel)
    if gamma == "None":
        gamma = 1

    # define scale bars
    scaleBarLength = getScaleBarLength(root, channel.voxelDims)
    scaleBarXY = scaleBar(
        posY = movieHeight,
        posX = lenX,
        length = scaleBarLength,
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXY._setFont()
    scaleBarXZ = scaleBar(
        posY = scaledLenZ,
        posX = lenX + gap + scaledLenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXZ._setFont()
    
    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try: 
        for i in tqdm(range(lenT)):

            # initialize frame 
            im = np.zeros([movieHeight,movieWidth])

            # copy max projections 
            imXZ = copy.copy(np.flip(maxY[i,nChannel],axis=0))
            im[0:scaledLenZ,0:lenX] = scaleXZYZ(imXZ, zToXYRatio)
            im[(scaledLenZ+gap):movieHeight,0:lenX] = copy.copy(maxZ[i,nChannel,0])
            imYZ = copy.copy(maxX[i,nChannel])
            im[(scaledLenZ+gap):movieHeight,(lenX+gap):movieWidth] = np.transpose(scaleXZYZ(imYZ, zToXYRatio))
            
            contrastedIm = adjustContrast(im, scaleMax, scaleMin, gamma)

            # invert if rock channel
            if channel.invertChannel:
                contrastedIm = 255 - contrastedIm

            frame = cv2.applyColorMap(contrastedIm,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = calculateTimeStamp(i, imagingFreq, lenT)
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def process_single_slice(slice_data, tiles_y, tiles_x):
    """Process a single 2D slice by normalizing each tile separately."""
    height, width = slice_data.shape
    normalized_slice = np.zeros_like(slice_data)
    
    tile_height = height // tiles_y
    tile_width = width // tiles_x
    
    # Process each tile independently
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Get tile boundaries
            y_start = i * tile_height
            y_end = min((i + 1) * tile_height, height)
            x_start = j * tile_width
            x_end = min((j + 1) * tile_width, width)
            
            # Extract and normalize this tile
            tile = slice_data[y_start:y_end, x_start:x_end]
            tile_mean = np.mean(tile)
            
            if np.abs(tile_mean) > 1e-6:  # Only normalize if tile isn't uniform
                normalized_tile = tile - tile_mean
                
                # Enhance contrast using percentile clipping
                p_low, p_high = np.percentile(normalized_tile, [5, 95])
                if p_high > p_low:
                    normalized_tile = np.clip(normalized_tile, p_low, p_high)
                    normalized_tile = (normalized_tile - p_low) / (p_high - p_low)
                else:
                    normalized_tile = np.zeros_like(tile)
            else:
                normalized_tile = np.zeros_like(tile)
            
            # Store the normalized tile back
            normalized_slice[y_start:y_end, x_start:x_end] = normalized_tile
    
    return normalized_slice

def normalize_vz_with_tiling(vz_data, tiles_y=4, tiles_x=3, sigma_smooth=3, sigma_blend=1):
    """
    Normalize z-flow data using per-tile mean subtraction to eliminate tiling artifacts.
    
    This function helps remove systematic errors that appear as grid patterns
    in the z-component of optical flow data.
    """
    # Smooth the data first to reduce noise
    vz_smoothed = gaussian_filter(vz_data, sigma=sigma_smooth)
    
    # Handle both 2D and 3D data
    if len(vz_data.shape) == 3:
        depth, height, width = vz_smoothed.shape
        vz_normalized = np.zeros_like(vz_smoothed)
        
        for d in range(depth):
            vz_normalized[d] = process_single_slice(vz_smoothed[d], tiles_y, tiles_x)
    else:
        vz_normalized = process_single_slice(vz_smoothed, tiles_y, tiles_x)
    
    # Smooth tile boundaries to blend them together
    vz_normalized = gaussian_filter(vz_normalized, sigma=sigma_blend)
    return vz_normalized


def enhance_channel_contrast(channel):
    """Enhance the contrast of a single channel using percentile normalization."""
    p_low, p_high = np.percentile(channel, [5, 95])
    if p_high > p_low:
        enhanced = np.clip(channel, p_low, p_high)
        enhanced = (enhanced - p_low) / (p_high - p_low)
        enhanced = enhanced ** 1.6  # Slight gamma correction to reduce brightness
        return enhanced
    return channel

def balance_rgb_channels(red, green, blue):
    """Balance RGB channels to prevent any single color from dominating."""
    # Calculate mean intensity for each channel
    red_mean = np.mean(red)
    green_mean = np.mean(green) 
    blue_mean = np.mean(blue)
    max_mean = max(red_mean, green_mean, blue_mean)
    
    if max_mean > 1e-6:
        # Scale channels relative to the brightest one
        if red_mean > 1e-6:
            red = red * (max_mean / red_mean) * 0.9    # Reduce red slightly
        if green_mean > 1e-6:
            green = green * (max_mean / green_mean) * 1.1  # Boost green slightly
        if blue_mean > 1e-6:
            blue = blue * (max_mean / blue_mean) * 0.6     # Reduce blue significantly
    
    return red, green, blue

def smooth_image(image, sigma=2.5, preserve_edges=True, passes=3):
    """
    Apply smoothing to improve visualization quality without resizing.
    Uses Gaussian filtering with optional edge preservation. Iterates over passes to improve smoothing 
    
    Args:
        image: Input image (grayscale or color)
        sigma: Smoothing strength (higher = more smoothing)
        preserve_edges: If True, uses edge-preserving filtering
    
    Returns:
        Smoothed image of same dimensions as input
    """
    # Handle different image types
    smoothed = image.copy()
    
    for _ in range(passes):
        if len(smoothed.shape) == 3:  # Color image
            if preserve_edges and cv2.__version__ >= '3.0.0':
                smoothed = cv2.bilateralFilter(smoothed.astype(np.float32), 
                                              d=0, 
                                              sigmaColor=sigma*15, 
                                              sigmaSpace=sigma)
        else:
            # Apply Gaussian smoothing to each channel
            smoothed = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                smoothed[:,:,c] = gaussian_filter(image[:,:,c], sigma=sigma)
    else:  # Grayscale
        if preserve_edges and cv2.__version__ >= '3.0.0':
            # Edge-preserving filter for grayscale images
            smoothed = cv2.bilateralFilter(image.astype(np.float32), 
                                           d=0,  # Automatic diameter
                                           sigmaColor=sigma*15, 
                                           sigmaSpace=sigma)
        else:
            # Standard Gaussian smoothing
            smoothed = gaussian_filter(image, sigma=sigma)
    
    # Ensure output is in same format as input
    if image.dtype == np.uint8:
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    
    return smoothed

def makeOrthoMaxOpticalFlowVideo(root, channel, ext='.mp4'):
    """
    Create a video showing optical flow in three orthogonal projections.
    
    Layout:
    - Top: YZ projection (side view)
    - Bottom-left: XY projection (top-down view)  
    - Bottom-right: XZ projection (front view)
    """
    # Generate unique filename
    filename = generateUniqueFilename(channel.name + '_optical_flow_orthomax_', ext)
    voxelDims = channel.voxelDims

    # Get imaging frequency for timestamps
    try:
        imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')
    except:
        imagingFreq = 10

    # Load the optical flow data
    flowMaxZ = root['flow_maxz']  # (T,4,Y,X) - XY projections over time
    flowMaxY = root['flow_maxy']  # (T,4,Z,X) - XZ projections over time  
    flowMaxX = root['flow_maxx']  # (T,4,Z,Y) - YZ projections over time
    
    print(f"Flow data dimensions:")
    print(f"  XY projection: {flowMaxZ.shape}")
    print(f"  XZ projection: {flowMaxY.shape}")
    print(f"  YZ projection: {flowMaxX.shape}")
    
    # Get dimensions
    lenT = flowMaxZ.shape[0]
    lenZ = flowMaxX.shape[2]
    lenY = flowMaxZ.shape[2] 
    lenX = flowMaxZ.shape[3]

    # Calculate total video dimensions
    gap = 20  # Space between projections
    movieWidth = lenX + gap + lenZ
    movieHeight = lenZ + gap + lenY
    
    print(f"Video layout: {movieWidth}x{movieHeight}")

    # Setup scale bar
    scaleBarLengths = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000]
    projDimsUm = [lenX*voxelDims[0], lenY*voxelDims[1], lenZ*voxelDims[2]]
    approxScaleBarLength = projDimsUm[1]/5
    scaleBarLength = min(scaleBarLengths, key=lambda x:abs(x-approxScaleBarLength))

    scaleBarXY = scaleBar(
        posY = movieHeight - 10,
        posX = lenX - 10, 
        length = scaleBarLength,
        pxPerMicron = 1/voxelDims[0],
        font = None,
    )
    scaleBarXY._setFont()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext.lower() == '.mp4' else cv2.VideoWriter_fourcc(*'MJPG')
    vid = cv2.VideoWriter(filename, fourcc, 10, (movieWidth, movieHeight), 1)
    
    try: 
        for i in tqdm(range(lenT), desc="Creating video frames"):
            # Initialize empty frame
            frame = np.zeros([movieHeight, movieWidth, 3], dtype=np.uint8)

            # Extract RGB channels for each projection at this time point
            red_xy, green_xy, blue_xy = flowMaxZ[i, 0], flowMaxZ[i, 1], flowMaxZ[i, 2]
            red_xz, green_xz, blue_xz = flowMaxY[i, 0], flowMaxY[i, 1], flowMaxY[i, 2] 
            red_yz, green_yz, blue_yz = flowMaxX[i, 0], flowMaxX[i, 1], flowMaxX[i, 2]
            
            # Process XY projection (standard contrast enhancement)
            red_xy = enhance_channel_contrast(red_xy)
            green_xy = enhance_channel_contrast(green_xy)
            blue_xy = enhance_channel_contrast(blue_xy)
            
            # Process XZ projection (with z-component normalization for blue channel)
            red_xz = enhance_channel_contrast(red_xz)
            green_xz = enhance_channel_contrast(green_xz)
            blue_xz = normalize_vz_with_tiling(blue_xz)  # Special processing for z-flow
            red_xz, green_xz, blue_xz = balance_rgb_channels(red_xz, green_xz, blue_xz)
            blue_xz = blue_xz * 0.7  # Further reduce blue dominance
            
            # Process YZ projection (with z-component normalization for blue channel)
            red_yz = enhance_channel_contrast(red_yz)
            green_yz = enhance_channel_contrast(green_yz)
            blue_yz = normalize_vz_with_tiling(blue_yz)  # Special processing for z-flow
            red_yz, green_yz, blue_yz = balance_rgb_channels(red_yz, green_yz, blue_yz)
            blue_yz = blue_yz * 0.7  # Further reduce blue dominance
            
            # Convert to 8-bit values
            brightness = 1.0
            def to_uint8(channel):
                return np.clip(channel * 255 * brightness, 0, 255).astype(np.uint8)
            
            # Convert all channels
            red_xy, green_xy, blue_xy = to_uint8(red_xy), to_uint8(green_xy), to_uint8(blue_xy)
            red_xz, green_xz, blue_xz = to_uint8(red_xz), to_uint8(green_xz), to_uint8(blue_xz)
            red_yz, green_yz, blue_yz = to_uint8(red_yz), to_uint8(green_yz), to_uint8(blue_yz)

            # Create RGB images (BGR format for OpenCV)
            rgb_xy = np.stack([blue_xy, green_xy, red_xy], axis=-1)
            rgb_xz = np.stack([blue_xz, green_xz, red_xz], axis=-1)
            rgb_yz = np.stack([blue_yz, green_yz, red_yz], axis=-1)

            # Place XY projection (bottom-left, main view)
            try:
                smoothed_xy = smooth_image(rgb_xy)
                y_start = lenZ + gap
                frame[y_start:y_start + lenY, 0:lenX] = smoothed_xy
            except Exception as e:
                print(f"Error placing XY projection: {e}")

            # Place XZ projection (bottom-right strip)
            try:
                # Flip vertically so Z increases upward
                rgb_xz_flipped = np.flip(rgb_xz, axis=0)
                smoothed_xz = smooth_image(rgb_xz_flipped)
                frame[0:lenZ, 0:lenX] = smoothed_xz
            except Exception as e:
                print(f"Error placing XZ projection: {e}")

            # Place YZ projection (top strip)
            try:
                # Transpose from (Z,Y,3) to (Y,Z,3) for proper orientation
                rgb_yz_transposed = np.transpose(rgb_yz, (1, 0, 2))
                smoothed_yz = smooth_image(rgb_yz_transposed)
                y_start = lenZ + gap
                x_start = lenX + gap
                frame[y_start:y_start + lenY, x_start:x_start + lenZ] = smoothed_yz
            except Exception as e:
                print(f"Error placing YZ projection: {e}")

            # Add scale bar and timestamp
            frame = scaleBarXY._addScaleBar(frame)
            t = calculateTimeStamp(i, imagingFreq, lenT)
            frame = addTimeStamp(frame, (10, 20), t, scaleBarXY.font)

            # Write frame to video
            vid.write(frame)

        # Cleanup
        vid.release()
        cv2.destroyAllWindows()
        print(f"✓ Video saved: {filename}")

    except Exception as e:
        print(f"Error creating video: {e}")
        traceback.print_exc()
        try:
            vid.release()
        except:
            pass
        cv2.destroyAllWindows()

def makeSlicedOrthoMaxVideos(root, channel, dim, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + "_" + dim + '_sliced_orthomax_' + cmap, ext)
    nChannel = channel.nChannel
    scaleMax = channel.scaleMax
    scaleMin = channel.scaleMin
    gamma = channel.gamma

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')
    
    slicedMax = root['sliced_max'+dim]

    lenT, lenZ, _, _ = getProjectionDimensions(root)

    gap = 20

    # calc scaleMin, scaleMax, and gamma if not provided
    if scaleMin == "None" or scaleMax == "None":
        scaleMin, scaleMax = calcAutoContrast(slicedMax, nChannel)
    if gamma == "None":
        gamma = 1 
        
    nSlices = slicedMax.shape[2]

    movieWidth = slicedMax.shape[-1]
    movieHeight = (lenZ * nSlices) + (gap * (nSlices-1))

    # define scale bars
    scaleBarLength = getScaleBarLength(root, channel.voxelDims)
    scaleBarXY = scaleBar(
        posY = movieHeight,
        posX = movieWidth,
        length = scaleBarLength,
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXY._setFont()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try:
        for i in tqdm(range(lenT)):

            # initialize frame
            im = np.zeros([movieHeight,movieWidth])

            # copy max projections 
            for j in range(nSlices):
                im[(lenZ*j+gap*j):(lenZ*(j+1)+gap*j),:] = copy.copy(np.flip(slicedMax[i,nChannel,j], axis=0))

            # adjust contrast
            contrastedIm = adjustContrast(im, scaleMax, scaleMin, gamma)

            # invert if rock channel
            if channel.invertChannel:
                contrastedIm = 255 - contrastedIm

            frame = cv2.applyColorMap(contrastedIm,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # add scale bar
            frame = scaleBarXY._addScaleBar(frame)

            # time stamp
            t = calculateTimeStamp(i, imagingFreq, lenT)
            timeStampPos = (0, 0)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def makeCompOrthoMaxVideo(root, channels, ext='.avi'):

    filename = generateUniqueFilename('comp_orthomax', ext)

    voxelDims = channels[0].voxelDims

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    maxZ = root['maxz']
    maxY = root['maxy']
    maxX = root['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)

    # calc scaled Z dimension
    zToXYRatio = voxelDims[2]/voxelDims[0]
    scaledLenZ = int(round(lenZ*zToXYRatio))
    
    gap = 20

    movieWidth = lenX + scaledLenZ + gap
    movieHeight = lenY + scaledLenZ + gap

    # calc scaleMin, scaleMax, and gamma if not provided
    for channel in channels:
        if channel.scaleMin == "None" or channel.scaleMax == "None":
            channel.scaleMin, channel.scaleMax = calcAutoContrast(maxZ, channel.nChannel)
        if channel.gamma == "None":
            channel.gamma = 1

    # define scale bars
    scaleBarLength = getScaleBarLength(root, channel.voxelDims)
    scaleBarXY = scaleBar(
        posY = movieHeight,
        posX = lenX,
        length = scaleBarLength,
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXY._setFont()
    scaleBarXZ = scaleBar(
        posY = scaledLenZ,
        posX = lenX + gap + scaledLenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXZ._setFont()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)
    
    try:
        for i in tqdm(range(lenT)):
            for channel in channels:
                im = np.zeros([movieHeight,movieWidth])

                # copy max projections 
                imXZ = copy.copy(np.flip(maxY[i,channel.nChannel],axis=0))
                im[0:scaledLenZ,0:lenX] = scaleXZYZ(imXZ, zToXYRatio)
                im[(scaledLenZ+gap):movieHeight,0:lenX] = copy.copy(maxZ[i,channel.nChannel,0])
                imYZ = copy.copy(maxX[i,channel.nChannel])
                im[(scaledLenZ+gap):movieHeight,(lenX+gap):movieWidth] = np.transpose(scaleXZYZ(imYZ, zToXYRatio))
                
                contrastedIm = adjustContrast(im, channel.scaleMax, channel.scaleMin, channel.gamma)

                # invert if rock channel
                if channel.invertChannel:
                    contrastedIm = 255 - contrastedIm

                # apply color map
                if channel.name == 'rocks':
                    greenIm = contrastedIm
                else:
                    purpleIm = contrastedIm

            # merge images
            frame = cv2.merge((purpleIm,greenIm,purpleIm))
            frame[np.where(im==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = calculateTimeStamp(i, imagingFreq, lenT)
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def generateZDepthColormap(lenZ, cmap):
    #generates a colormap based on z depth, red is the highest z depth, blue is the lowest
    zDepthColormap = [None]*lenZ
    for slice in range(0,lenZ):
        zDepthGrayVal = round((slice/lenZ)*255)
        zDepthColormap[slice] = cmapy.color(cmap, zDepthGrayVal)
    return zDepthColormap

def invertAndScale(im, invert):
    # invert if rock channel
    if invert:
        im = 255 - im
    scaledIm = np.divide(im,255)
    scaledImGrayscale = cv2.merge([scaledIm, scaledIm, scaledIm])
    return scaledImGrayscale

def makeZDepthOrthoMaxVideo(root, channel, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + '_zdepth_orthomax_' + cmap, ext)
    nChannel = channel.nChannel
    voxelDims = channel.voxelDims
    scaleMax = channel.scaleMax
    scaleMin = channel.scaleMin
    gamma = channel.gamma

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    maxZ = root['maxz']
    maxY = root['maxy']
    maxX = root['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)
    
    # calc scaled Z dimension
    zToXYRatio = voxelDims[2]/voxelDims[0]
    scaledLenZ = int(round(lenZ*zToXYRatio))

    zDepthColormap = generateZDepthColormap(lenZ, cmap)
    zDepthColormapXZYZ = generateZDepthColormap(scaledLenZ, cmap)

    gap = 20

    movieWidth = lenX + scaledLenZ + gap
    movieHeight = lenY + scaledLenZ + gap

    # calc scaleMin, scaleMax, and gamma if not provided
    if scaleMin == "None" or scaleMax == "None":
        scaleMin, scaleMax = calcAutoContrast(maxZ, nChannel)
    if gamma == "None":
        gamma = 1

    # define scale bars
    scaleBarLength = getScaleBarLength(root, channel.voxelDims)
    scaleBarXY = scaleBar(
        posY = movieHeight,
        posX = lenX,
        length = scaleBarLength,
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXY._setFont()
    scaleBarXZ = scaleBar(
        posY = scaledLenZ,
        posX = lenX + gap + scaledLenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[0],
        font = None,
    )
    scaleBarXZ._setFont()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try:
        for i in tqdm(range(lenT)):
            # generate a scaled image for the XY projection
            imXY = copy.copy(maxZ[i,nChannel,0])
            contrastedImXY = adjustContrast(imXY, scaleMax, scaleMin, gamma)
            scaledImGrayscaleXY = invertAndScale(contrastedImXY, channel.invertChannel)
            # apply z depth colormap based on z depths in slice
            zDepths = maxZ[i,nChannel,1]
            # TODO: make into functions for XY color assignment and XZ/YZ color assignment
            imBGRValsXY = np.zeros([lenY, lenX, 3]).astype(int)
            for y in range(0,lenY):
                for x in range(0,lenX):
                    zDepth = int(zDepths[y,x])
                    imBGRValsXY[y,x,:] = zDepthColormap[zDepth]
            frameXY = np.multiply(scaledImGrayscaleXY,imBGRValsXY).astype('uint8')

            # generate a scaled image for the XZ projection
            imXZ = copy.copy(maxY[i,nChannel])
            imXZ = scaleXZYZ(imXZ, zToXYRatio)

            contrastedImXZ = adjustContrast(imXZ, scaleMax, scaleMin, gamma)
            scaledImGrayscaleXZ = invertAndScale(contrastedImXZ, channel.invertChannel)

            # apply z depth colormap based on z depths in slice
            imBGRValsXZ = np.zeros([scaledLenZ, lenX, 3]).astype(int)
            for z in range(0,scaledLenZ):
                zDepth = z
                imBGRValsXZ[z,:,:] = zDepthColormapXZYZ[zDepth]
            frameXZ = np.multiply(scaledImGrayscaleXZ,imBGRValsXZ).astype('uint8')
            frameXZ = np.flip(frameXZ, axis=0)

            # generate a scaled image for the YZ projection
            imYZ = copy.copy(maxX[i,nChannel])
            imYZ = scaleXZYZ(imYZ, zToXYRatio)
            contrastedImYZ = adjustContrast(imYZ, scaleMax, scaleMin, gamma)
            scaledImGrayscaleYZ = invertAndScale(contrastedImYZ, channel.invertChannel)

            # apply z depth colormap based on z depths in slice
            imBGRValsYZ = np.zeros([scaledLenZ, lenY, 3]).astype(int)
            for z in range(0,scaledLenZ):
                zDepth = z
                imBGRValsYZ[z,:,:] = zDepthColormapXZYZ[zDepth]
            frameYZ = np.multiply(scaledImGrayscaleYZ,imBGRValsYZ).astype('uint8')
            frameYZ = np.transpose(frameYZ, (1,0,2))

            # initialize frame 
            frame = np.zeros([movieHeight,movieWidth,3]).astype('uint8')

            frame[0:scaledLenZ,0:lenX,:] = frameXZ
            frame[(scaledLenZ+gap):movieHeight,0:lenX,:] = frameXY
            frame[(scaledLenZ+gap):movieHeight,(lenX+gap):movieWidth,:] = frameYZ

            #frame[np.where(scaledIm==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = calculateTimeStamp(i, imagingFreq, lenT)
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()