# Dicty data functions for ome-zarr datasets

import copy
import os

import xml.etree.ElementTree as et
import zarr
import cv2
import cmapy
import json
from PIL import ImageFont, ImageDraw, Image
from dask.distributed import Client, wait
import numpy as np
from tqdm import tqdm


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

def calcOpticalFlowMaxProjections(maxProjectionsRoot):

    """Create max projections specifically from optical flow RGBM (rgb and magnitude) arrays"""

    flow_results_dir = '/Volumes/sgrolab/jennifer/cryolite/cryolite_mixin_test53_2025-02-22/optical_flow_3Dresults'

    #if cropping, get crop parameters from parameters.json and redefine resArray
    cropDims = getCroppingDimsFromJSON(maxProjectionsRoot.store.path + '/../../parameters.json')
    if cropDims:
        resArray = resArray[:, :, cropDims[2][0]:cropDims[2][1], cropDims[1][0]:cropDims[1][1], cropDims[0][0]:cropDims[0][1]]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    # Create zarr arrays for optical flow RGBM max projections
    flowMaxZ = maxProjectionsRoot.zeros('flow_maxz', shape=(lenT, 4, lenY, lenX), chunks=(1, 4, lenY, lenX))
    flowMaxX = maxProjectionsRoot.zeros('flow_maxx', shape=(lenT, 4, lenZ, lenY), chunks=(1, 4, lenZ, lenY))
    flowMaxY = maxProjectionsRoot.zeros('flow_maxy', shape=(lenT, 4, lenZ, lenX), chunks=(1, 4, lenZ, lenX))

    # iterate through each timepoint and compute max projections
    for i, frame_num in enumerate(tqdm(lenT)):
        frame_dir = os.path.join(flow_results_dir, str(frame_num))
        rgbm_file = os.path.join(frame_dir, "optical_flow_rgbm.npy")
        
        if os.path.exists(rgbm_file):
            # Load RGBM data: (Z, Y, X, 4) where 4 = [Red, Green, Blue, Magnitude]
            rgbm_data = np.load(rgbm_file)

        for j in range(4):
            channel_data = rgbm_data[:, :, :, j]  # (Z, Y, X)
    
            flowMaxZ[i, j] = np.max(channel_data, axis=0)  # (Y, X)
            flowMaxX[i, j] = np.max(channel_data, axis=2)  # (Z, Y) 
            flowMaxY[i, j] = np.max(channel_data, axis=1)  # (Z, X)

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
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def makeOrthoMaxOpticalFlowVideo(root, channel, ext='.mp4'):

    """Generates an optical flow orthomax video"""

    filename = generateUniqueFilename(channel.name + '_optical_flow_orthomax_', ext)
    voxelDims = channel.voxelDims

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/../../parameters.json')

    flowMaxZ = root['flow_maxz'] # (T,4,Y,X)
    flowMaxY = root['flow_maxy']
    flowMaxX = root['flow_maxx']
    
    lenT = flowMaxZ.shape[0]
    lenZ = flowMaxX.shape[2]
    lenY = flowMaxZ.shape[2]
    lenX = flowMaxZ.shape[3]

    # calc scaled Z dimension
    zToXYRatio = voxelDims[2]/voxelDims[0]
    scaledLenZ = int(round(lenZ*zToXYRatio))
    
    gap = 20

    movieWidth = lenX + scaledLenZ + gap
    movieHeight = lenY + scaledLenZ + gap

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

            # Initialize RGB frame 
            frame = np.zeros([movieHeight, movieWidth, 3], dtype=np.uint8)

            # Extract RGBM channels for this timepoint
            red_xy = flowMaxZ[i, 0]     # (Y, X)
            green_xy = flowMaxZ[i, 1]   # (Y, X) 
            blue_xy = flowMaxZ[i, 2]    # (Y, X)
            
            red_xz = flowMaxY[i, 0]     # (Z, X)
            green_xz = flowMaxY[i, 1]   # (Z, X)
            blue_xz = flowMaxY[i, 2]    # (Z, X)
            
            red_yz = flowMaxX[i, 0]     # (Z, Y)
            green_yz = flowMaxX[i, 1]   # (Z, Y)
            blue_yz = flowMaxX[i, 2]    # (Z, Y)

            # Scale values from [0,1] to [0,255] for display
            red_xy = np.clip(red_xy * 255, 0, 255).astype(np.uint8)
            green_xy = np.clip(green_xy * 255, 0, 255).astype(np.uint8)
            blue_xy = np.clip(blue_xy * 255, 0, 255).astype(np.uint8)
            
            red_xz = np.clip(red_xz * 255, 0, 255).astype(np.uint8)
            green_xz = np.clip(green_xz * 255, 0, 255).astype(np.uint8)
            blue_xz = np.clip(blue_xz * 255, 0, 255).astype(np.uint8)
            
            red_yz = np.clip(red_yz * 255, 0, 255).astype(np.uint8)
            green_yz = np.clip(green_yz * 255, 0, 255).astype(np.uint8)
            blue_yz = np.clip(blue_yz * 255, 0, 255).astype(np.uint8)

            # Create RGB images for each projection
            rgb_xy = np.stack([blue_xy, green_xy, red_xy], axis=-1)  # BGR for OpenCV
            frame[(scaledLenZ+gap):movieHeight, 0:lenX] = rgb_xy

            # XZ projection (top) - scale and flip 
            rgb_xz = np.stack([blue_xz, green_xz, red_xz], axis=-1)
            rgb_xz_flipped = np.flip(rgb_xz, axis=0)
            rgb_xz_scaled = np.zeros((scaledLenZ, lenX, 3), dtype=np.uint8)
            
            for z in range(lenZ):
                z_start = int(z * zToXYRatio)
                z_end = min(int((z + 1) * zToXYRatio), scaledLenZ)
                if z_start < scaledLenZ:
                    rgb_xz_scaled[z_start:z_end, :] = rgb_xz_flipped[z]
            
            frame[0:scaledLenZ, 0:lenX] = rgb_xz_scaled

            # YZ projection (bottom right) - scale and transpose
            rgb_yz = np.stack([blue_yz, green_yz, red_yz], axis=-1)
            rgb_yz_scaled = np.zeros((scaledLenZ, lenY, 3), dtype=np.uint8)
            
            for z in range(lenZ):
                z_start = int(z * zToXYRatio)
                z_end = min(int((z + 1) * zToXYRatio), scaledLenZ)
                if z_start < scaledLenZ:
                    rgb_yz_scaled[z_start:z_end, :] = rgb_yz[z]
            
            rgb_yz_transposed = np.transpose(rgb_yz_scaled, (1, 0, 2))
            frame[(scaledLenZ+gap):movieHeight, (lenX+gap):movieWidth] = rgb_yz_transposed

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        # Close video after all frames are written
        vid.release()
        cv2.destroyAllWindows()
        print(f"✓ Optical flow orthomax video saved: {filename}")

    except Exception as e:
        print(f"Error creating optical flow video: {e}")
        vid.release()
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
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
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
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
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
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, scaledLenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()