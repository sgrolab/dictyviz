# Dicty data functions for ome-zarr datasets

import copy
import os

import xml.etree.ElementTree as et
import zarr
import cv2
import cmapy
import json
from PIL import ImageFont, ImageDraw, Image
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
    def __init__(self, name, nChannel, voxelDims, scaleMax, scaleMin=0):
        self.name = name
        self.nChannel = nChannel
        self.voxelDims = voxelDims
        self.scaleMax = scaleMax
        self.scaleMin = scaleMin

def getChannelsFromJSON(jsonFile):
    with open(jsonFile) as f:
        channelSpecs = json.load(f)["channels"]
    channels = []
    for channelInfo in channelSpecs:
        channels.append(channel(name=channelInfo["name"],
                                nChannel=channelInfo["channelNumber"],
                                voxelDims=None,
                                scaleMax=channelInfo["scaleMax"],
                                scaleMin=channelInfo["scaleMin"]))
    return channels

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


def calcMaxProjections(root, analysisRoot, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    analysisGroup = analysisRoot['analysis']

    # create max projections group
    maxProjectionsGroup = createZarrGroup(analysisGroup, 'max_projections')

    # create zarr arrays for each max projection 
    maxZ = maxProjectionsGroup.zeros('maxz',shape=(lenT,lenCh,2,lenY,lenX),chunks=(1,lenCh,2,lenY,lenX))
    maxX = maxProjectionsGroup.zeros('maxx',shape=(lenT,lenCh,lenZ,lenY),chunks=(1,lenCh,lenZ,lenY))
    maxY = maxProjectionsGroup.zeros('maxy',shape=(lenT,lenCh,lenZ,lenX),chunks=(1,lenCh,lenZ,lenX))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            frame = resArray[i, j, :, :, :]
            maxZ[i,j] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
            maxX[i,j] = np.max(frame,axis=2)
            maxY[i,j] = np.max(frame,axis=1)          


def calcSlicedMaxProjections(root, analysisRoot, res_lvl=0):
    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = resArray.shape

    analysisGroup = analysisRoot['analysis']

    # create max projections group
    slicedMaxProjectionsGroup = createZarrGroup(analysisGroup, 'sliced_max_projections')

    # set number of slices
    #sliceDepth = 83 # 83px*2.41um/px = 200 um
    nSlices = 20
    sliceDepthX = lenX//(nSlices-1)
    sliceDepthY = lenY//(nSlices-1)
    #nSlicesX = math.ceil(lenX/sliceDepth)
    #nSlicesY = math.ceil(lenY/sliceDepth)

    # create zarr arrays for each max projection
    slicedMaxX = slicedMaxProjectionsGroup.zeros('sliced_maxx',shape=(lenT,lenCh,nSlices,lenZ,lenY),chunks=(1,1,2,lenZ,lenY))
    slicedMaxY = slicedMaxProjectionsGroup.zeros('sliced_maxy',shape=(lenT,lenCh,nSlices,lenZ,lenX),chunks=(1,1,2,lenZ,lenX))

    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            for k in range(nSlices-1):
                rangeX = [k*sliceDepthX, (k+1)*sliceDepthX]
                frame = resArray[i,j,:,:,rangeX[0]:rangeX[1]]
                slicedMaxX[i,j,k] = np.max(frame,axis=2)
            #fill last chunk with the rest of the data
            rangeX = [(nSlices-1)*sliceDepthX, lenX]
            frame = resArray[i,j,:,:,rangeX[0]:rangeX[1]]
            slicedMaxX[i,j,nSlices-1] = np.max(frame,axis=2)

            for k in range(nSlices-1):
                rangeY = [k*sliceDepthY, (k+1)*sliceDepthY]
                frame = resArray[i,j,:,rangeY[0]:rangeY[1],:]
                slicedMaxY[i,j,k] = np.max(frame,axis=1)
            #fill last chunk with the rest of the data
            rangeY = [(nSlices-1)*sliceDepthY, lenY]
            frame = resArray[i,j,:,rangeY[0]:rangeY[1],:]
            slicedMaxY[i,j,nSlices-1] = np.max(frame,axis=1)

def generateUniqueFilename(filename, ext):
    i = 1
    while os.path.exists(filename + ext):
        if i == 1:
            filename = filename +'_1'
        else:
            filename = filename[:-1] + str(i)
        i += 1
    return filename + ext

# replace with an adjustable auto contrast of some sort
def calcScaleMax(root):
    maxZ = root['analysis']['max_projections']['maxz']
    scaleMax = np.max(maxZ)
    return scaleMax

def getProjectionDimensions(root):
    # return the dimensions of the max projections
    maxX = root['analysis']['max_projections']['maxx']
    maxY = root['analysis']['max_projections']['maxy']
    lenT = maxX.shape[0]
    lenZ = maxX.shape[-2]
    lenY = maxX.shape[-1]
    lenX = maxY.shape[-1]
    return lenT, lenZ, lenY, lenX

def adjustContrast(im, adjMax, scaleMin):
    im = np.clip(im,scaleMin,adjMax)
    backSub = im - scaleMin
    backSub[np.where(backSub<0)] = 0
    scaledIm = np.divide(backSub,adjMax-scaleMin)
    contrastedIm = np.multiply(scaledIm,255).astype('uint8')
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


def makeOrthoMaxVideo(root, channel, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + '_orthomax_' + cmap, ext)
    nChannel = channel.nChannel
    adjMax = channel.scaleMax
    scaleMin = channel.scaleMin

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/parameters.json')

    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)
    
    gap = 20

    movieWidth = lenX + lenZ + gap
    movieHeight = lenY + lenZ + gap

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
        posY = lenZ,
        posX = lenX + gap + lenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[2],
        font = None,
    )
    scaleBarXZ._setFont()
    
    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try: 
        for i in tqdm(range(lenT)):

            # initialize frame 
            im = np.zeros([movieHeight,movieWidth])

            # copy max projections 
            im[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,nChannel],axis=0))
            im[(lenZ+gap):movieHeight,0:lenX] = copy.copy(maxZ[i,nChannel,0])
            im[(lenZ+gap):movieHeight,(lenX+gap):movieWidth] = copy.copy(np.transpose(maxX[i,nChannel]))
            
            contrastedIm = adjustContrast(im, adjMax, scaleMin)

            # invert if rock channel
            if channel.name == 'rocks':
                contrastedIm = 255 - contrastedIm

            frame = cv2.applyColorMap(contrastedIm,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, lenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def makeSlicedOrthoMaxVideos(root, channel, cmap, ext='.avi'):

    filenames = [generateUniqueFilename(channel.name + '_X_sliced_orthomax_' + cmap, ext),
                 generateUniqueFilename(channel.name + '_Y_sliced_orthomax_' + cmap, ext)]
    nChannel = channel.nChannel
    adjMax = channel.scaleMax
    scaleMin = channel.scaleMin

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/parameters.json')
        
    slicedMaxes = [root['analysis']['sliced_max_projections']['sliced_maxx'], root['analysis']['sliced_max_projections']['sliced_maxy']]

    lenT, lenZ, _, _ = getProjectionDimensions(root)

    gap = 20

    for filename, slicedMax in zip(filenames, slicedMaxes):
        
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
                contrastedIm = adjustContrast(im, adjMax, scaleMin)

                # invert if rock channel
                if channel.name == 'rocks':
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

    # set channel values
    for channel in channels:
        if channel.name == 'cells':
            nChannelCells = channel.nChannel
            scaleMaxCells = channel.scaleMax
            scaleMinCells = channel.scaleMin
        else:
            nChannelRocks = channel.nChannel
            scaleMaxRocks = channel.scaleMax
            scaleMinRocks = channel.scaleMin

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/parameters.json')

    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)

    gap = 20

    movieWidth = lenX + lenZ + gap
    movieHeight = lenY + lenZ + gap

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
        posY = lenZ,
        posX = lenX + gap + lenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[2],
        font = None,
    )
    scaleBarXZ._setFont()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try:
        for i in tqdm(range(lenT)):
            
            imCells = np.zeros([movieHeight,movieWidth])
            imRocks = np.zeros([movieHeight,movieWidth])

            # copy max projections 
            imCells[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,nChannelCells],axis=0))
            imCells[(lenZ+gap):movieHeight,0:lenX] = copy.copy(maxZ[i,nChannelCells,0])
            imCells[(lenZ+gap):movieHeight,(lenX+gap):movieWidth] = copy.copy(np.transpose(maxX[i,nChannelCells]))

            imRocks[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,nChannelRocks],axis=0))
            imRocks[(lenZ+gap):movieHeight,0:lenX] = copy.copy(maxZ[i,nChannelRocks,0])
            imRocks[(lenZ+gap):movieHeight,(lenX+gap):movieWidth] = copy.copy(np.transpose(maxX[i,nChannelRocks]))
            
            contrastedImCells = adjustContrast(imCells, scaleMaxCells, scaleMinCells)
            contrastedImRocks = adjustContrast(imRocks, scaleMaxRocks, scaleMinRocks)

            # invert rock channel
            contrastedImRocks = 255 - contrastedImRocks

            frame = cv2.merge((contrastedImCells,contrastedImRocks,contrastedImCells))

            frame[np.where(contrastedImCells==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)
            
            # time stamp
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, lenZ+gap)
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

def invertAndScale(channel, im):
    # invert if rock channel
    if channel == 'rocks':
        im = 255 - im
    scaledIm = np.divide(im,255)
    scaledImGrayscale = cv2.merge([scaledIm, scaledIm, scaledIm])
    return scaledImGrayscale

def makeZDepthOrthoMaxVideo(root, channel, cmap, ext='.avi'):

    filename = generateUniqueFilename(channel.name + '_zdepth_orthomax_' + cmap, ext)
    nChannel = channel.nChannel
    adjMax = channel.scaleMax
    scaleMin = channel.scaleMin

    imagingFreq = getImagingFreqFromJSON(root.store.path + '/parameters.json')

    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    lenT, lenZ, lenY, lenX = getProjectionDimensions(root)

    zDepthColormap = generateZDepthColormap(lenZ, cmap)
    
    gap = 20

    movieWidth = lenX + lenZ + gap
    movieHeight = lenY + lenZ + gap

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
        posY = lenZ,
        posX = lenX + gap + lenZ,
        length = int(lenZ*channel.voxelDims[2]),
        pxPerMicron = 1/channel.voxelDims[2],
        font = None,
    )
    scaleBarXZ._setFont()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movieWidth,movieHeight),1)

    try:
        for i in tqdm(range(lenT)):

            # generate a scaled image for the XY projection
            imXY = copy.copy(maxZ[i,nChannel,0])
            contrastedImXY = adjustContrast(imXY, adjMax, scaleMin)
            scaledImGrayscaleXY = invertAndScale(channel.name, contrastedImXY)

            # apply z depth colormap based on z depths in slice
            zDepths = maxZ[i,nChannel,1]
            # TODO: make into functions for XY color assignment and XZ/YZ color assignment
            imBluesXY = np.zeros([lenY, lenX]).astype(int)
            imGreensXY = np.zeros([lenY, lenX]).astype(int)
            imRedsXY = np.zeros([lenY, lenX]).astype(int)
            for y in range(0,lenY):
                for x in range(0,lenX):
                    zDepth = int(zDepths[y,x])
                    imBluesXY[y,x] = zDepthColormap[zDepth][0]
                    imGreensXY[y,x] = zDepthColormap[zDepth][1]
                    imRedsXY[y,x] = zDepthColormap[zDepth][2]
            imBGRValsXY = cv2.merge([imBluesXY, imGreensXY, imRedsXY])

            frameXY = np.multiply(scaledImGrayscaleXY,imBGRValsXY).astype('uint8')


            # generate a scaled image for the XZ projection
            imXZ = copy.copy(maxY[i,nChannel])
            contrastedImXZ = adjustContrast(imXZ, adjMax, scaleMin)
            scaledImGrayscaleXZ = invertAndScale(channel.name, contrastedImXZ)

            # apply z depth colormap based on z depths in slice
            imBluesXZ = np.zeros([lenZ, lenX]).astype(int)
            imGreensXZ = np.zeros([lenZ, lenX]).astype(int)
            imRedsXZ = np.zeros([lenZ, lenX]).astype(int)
            for z in range(0,lenZ):
                for x in range(0,lenX):
                    zDepth = z
                    imBluesXZ[z,x] = zDepthColormap[zDepth][0]
                    imGreensXZ[z,x] = zDepthColormap[zDepth][1]
                    imRedsXZ[z,x] = zDepthColormap[zDepth][2]
            imBGRValsXZ = cv2.merge([imBluesXZ, imGreensXZ, imRedsXZ])

            frameXZ = np.multiply(scaledImGrayscaleXZ,imBGRValsXZ).astype('uint8')
            frameXZ = np.flip(frameXZ, axis=0)

            # generate a scaled image for the YZ projection
            imYZ = copy.copy(np.transpose(maxX[i,nChannel]))
            contrastedImYZ = adjustContrast(imYZ, adjMax, scaleMin)
            scaledImGrayscaleYZ = invertAndScale(channel.name, contrastedImYZ)

            # apply z depth colormap based on z depths in slice
            imBluesYZ = np.zeros([lenY, lenZ]).astype(int)
            imGreensYZ = np.zeros([lenY, lenZ]).astype(int)
            imRedsYZ = np.zeros([lenY, lenZ]).astype(int)
            for y in range(0,lenY):
                for z in range(0,lenZ):
                    zDepth = z
                    imBluesYZ[y,z] = zDepthColormap[zDepth][0]
                    imGreensYZ[y,z] = zDepthColormap[zDepth][1]
                    imRedsYZ[y,z] = zDepthColormap[zDepth][2]
            imBGRValsYZ = cv2.merge([imBluesYZ, imGreensYZ, imRedsYZ])

            frameYZ = np.multiply(scaledImGrayscaleYZ,imBGRValsYZ).astype('uint8')

            # initialize frame 
            frame = np.zeros([movieHeight,movieWidth,3]).astype('uint8')

            frame[0:lenZ,0:lenX,:] = frameXZ
            frame[(lenZ+gap):movieHeight,0:lenX,:] = frameXY
            frame[(lenZ+gap):movieHeight,(lenX+gap):movieWidth,:] = frameYZ

            #frame[np.where(scaledIm==0)] = [0,0,0]

            # add scale bars
            frame = scaleBarXY._addScaleBar(frame)
            frame = scaleBarXZ._addScaleBar(frame)

            # time stamp
            t = f'{(i*imagingFreq) // 60:02d}' +'hr:' + f'{(i*imagingFreq) % 60:02d}' + 'min'
            timeStampPos = (0, lenZ+gap)
            frame = addTimeStamp(frame, timeStampPos, t, scaleBarXY.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()