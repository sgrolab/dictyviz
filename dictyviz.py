# Dicty data functions for ome-zarr datasets

import os, zarr, cv2, cmapy, copy, math
import numpy as np
from tqdm import tqdm

# define global variables
IMAGING_FREQ = 10

def createRootStore(zarrFile):
    nestedStore = zarr.NestedDirectoryStore(zarrFile, dimension_separator='/')
    root = zarr.group(store=nestedStore, overwrite=False)

def createZarrGroup(root, groupName):

    if groupName in root:
        group = root[groupName]
    else:
        group = root.create_group(groupName)
    return group

def getDimensions(resArray):
    return resArray.shape[0], resArray.shape[1], resArray.shape[2], resArray.shape[3], resArray.shape[4]

def calcMaxProjections(root, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = getDimensions(resArray)
    
    analysisGroup = root['analysis']

    # create max projections group
    maxProjectionsGroup = createZarrGroup(analysisGroup, 'max_projections')

    # create zarr arrays for each max projection 
    maxZ = maxProjectionsGroup.zeros('maxz',shape=(lenT,lenCh,lenY,lenX),chunks=(1,lenCh,lenY,lenX))
    maxX = maxProjectionsGroup.zeros('maxx',shape=(lenT,lenCh,lenZ,lenY),chunks=(1,lenCh,lenZ,lenY))
    maxY = maxProjectionsGroup.zeros('maxy',shape=(lenT,lenCh,lenZ,lenX),chunks=(1,lenCh,lenZ,lenX))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(lenT)):
        for j in range(lenCh):
            frame = resArray[i, j, :, :, :]
            maxZ[i,j] = np.max(frame,axis=0)
            maxX[i,j] = np.max(frame,axis=2)
            maxY[i,j] = np.max(frame,axis=1)


def calcSlicedMaxProjections(root, res_lvl=0):
    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenCh, lenZ, lenY, lenX = getDimensions(resArray)

    analysisGroup = root['analysis']

    # create max projections group
    slicedMaxProjectionsGroup = createZarrGroup(analysisGroup, 'sliced_max_projections')

    # set number of slices
    sliceDepth = 83 # 83px*2.41um/px = 200 um
    nSlicesX = math.ceil(lenX/sliceDepth)
    nSlicesY = math.ceil(lenY/sliceDepth)

    # create zarr arrays for each max projection
    slicedMaxX = slicedMaxProjectionsGroup.zeros('sliced_maxx',shape=(nSlicesX,lenT,lenCh,lenZ,lenY),chunks=(1,1,1,64,64))
    slicedMaxY = slicedMaxProjectionsGroup.zeros('sliced_maxy',shape=(nSlicesY,lenT,lenCh,lenZ,lenX),chunks=(1,1,1,64,64))

    for i in range(nSlicesX):
        for j in range(lenT):
            for k in range(lenCh):
                rangeX = [i*sliceDepth, (i+1)*sliceDepth]
                frame = resArray[j, k, :, :, rangeX[0]:rangeX[1]]
                slicedMaxX[i,j,k] = np.max(frame,axis=2)
    
    for i in range(nSlicesY):
        for j in range(lenT):
            for k in range(lenCh):
                rangeY = [i*sliceDepth, (i+1)*sliceDepth]
                frame = resArray[j, k, :, rangeY[0]:rangeY[1], :]
                slicedMaxY[i,j,k] = np.max(frame,axis=1)

# replace with an adjustable auto contrast of some sort
def calcScaleMax(root):
    maxZ = root['analysis']['max_projections']['maxz']
    scaleMax = np.max(maxZ)
    return scaleMax

def adjustContrast(im, adjMax, adjMin):
    im[np.where(im>adjMax)] = adjMax
    im[np.where(im<adjMin)] = adjMin
    backSub = im - adjMin
    backSub[np.where(backSub<0)] = 0
    scaledIm = np.divide(backSub,adjMax-adjMin)
    contrastedIm = np.multiply(scaledIm,255).astype('uint8')
    return(contrastedIm)

class scaleBar:
    def __init__(self, posY, posX, height, length, text, textOffset):
        self.posY = posY
        self.posX = posX
        self.height = height
        self.length = length
        self.text = text
        self.textOffset = textOffset

    def _addScaleBar(self, frame):
        frame[self.posY:self.posY+self.height, self.posX:self.posX+self.length,:] = 255
        cv2.putText(frame, self.text, (self.posX+self.textOffset, self.posY-10), #self.posY-30
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 3, cv2.LINE_AA) #3, 6

    def _addScaleBarZ(self, frame):
        frame[self.posY:self.posY+self.height, self.posX:self.posX+self.length,:] = 255
        cv2.putText(frame, self.text, (self.posX, self.posY-10), #self.posY-50
                    cv2.FONT_HERSHEY_SIMPLEX, 0.1, [255,255,255], 1, cv2.LINE_AA)

def makeOrthoMaxVideo(filename, root, nChannel, scaleMax):

    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    lenT = maxZ.shape[0]
    lenZ = maxY.shape[2]
    lenY = maxZ.shape[2]
    lenX = maxZ.shape[3]
    
    gap = 20

    xz = lenX + lenZ + gap
    yz = lenY + lenZ + gap

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)

    #adjust contrast based on max pixel value in the array
    adjMax = scaleMax
    adjMin = 0

    for i in tqdm(range(lenT)):

        # initialize frame 
        im = np.zeros([yz,xz])

        # copy max projections 
        im[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,nChannel],axis=0))
        im[(lenZ+gap):yz,0:lenX] = copy.copy(maxZ[i,nChannel])
        im[(lenZ+gap):yz,(lenX+gap):xz] = copy.copy(np.transpose(maxX[i,nChannel]))
        
        contrastedIm = adjustContrast(im, adjMax, adjMin)
        frame = cv2.applyColorMap(contrastedIm,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]
        
        # time stamp
        t = f'{i*IMAGING_FREQ // 60:02d}' + ':' + f'{i*IMAGING_FREQ % 60:02d}'
        #cv2.putText(frame,t,(25,lenZ+gap+150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)
        cv2.putText(frame,t,(15,lenZ+gap+30),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,255],3,cv2.LINE_AA)

        # add scale bars
        scaleBarXY = scaleBar(
            posY = yz - 20, #76
            posX = lenX - 50, #468
            height = 10, #30
            length = 42, #416
            text = '100 um', #1 mm
            textOffset = 0, #50
        )
        scaleBarXY._addScaleBar(frame)

        scaleBarXZ = scaleBar(
            posY = lenZ,
            posX = lenX + gap,
            height = 10, #30
            length = lenZ,
            text = str(lenZ * 2) + ' um',
            textOffset = 5, #50
        )
        scaleBarXZ._addScaleBarZ(frame)

        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

def makeSlicedOrthoMaxVideos(filename, root, nChannel, scaleMax):
        
    slicedMaxX = root['analysis']['sliced_max_projections']['sliced_maxx']

    nSlices = slicedMaxX.shape[0]
    lenT = slicedMaxX.shape[1]
    lenZ = slicedMaxX.shape[2]
    lenY = slicedMaxX.shape[3]

    gap = 20

    sizeX = lenY
    sizeY = (lenZ * nSlices) + (gap * (nSlices-1))

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(sizeX,sizeY),1)

    adjMax = scaleMax
    adjMin = 0

    for i in tqdm(range(lenT)):

        # initialize frame
        im = np.zeros([sizeY,sizeX])

        # copy max projections 
        for j in range(nSlices):
            im[(lenZ*j+gap*j):(lenZ*(j+1)+gap*j),:] = copy.copy(np.flip(slicedMaxX[j,i,nChannel], axis=0))

        # adjust contrast
        contrastedIm = adjustContrast(im, adjMax, adjMin)
        frame = cv2.applyColorMap(contrastedIm,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]

        # add time stamp
        t = f'{i*IMAGING_FREQ // 60:02d}' + ':' + f'{i*IMAGING_FREQ % 60:02d}'
        cv2.putText(frame,t,(25,150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)

        # add scale bar
        scaleBarXY = scaleBar(
            posY = yz - 76,
            posX = lenX - 468,
            height = 30,
            length = 416,
            text = '1 mm',
            textOffset = 50,
        )
        scaleBarXY._addScaleBar(frame)

        # write frame
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

def makeCompOrthoMaxVideo(filename, root, scaleMax):
    
    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    lenT = maxZ.shape[0]
    lenZ = maxY.shape[2]
    lenY = maxZ.shape[2]
    lenX = maxZ.shape[3]
    
    gap = 20

    xz = lenX + lenZ + gap
    yz = lenY + lenZ + gap

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)

    #adjust contrast based on max pixel value in the array
    adjMax = scaleMax
    adjMin = 0

    for i in tqdm(range(lenT)):

        # initialize frame 
        imCh1 = np.zeros([yz,xz])
        imCh2 = np.zeros([yz,xz])

        # copy max projections 
        imCh1[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,0],axis=0))
        imCh1[(lenZ+gap):yz,0:lenX] = copy.copy(maxZ[i,0])
        imCh1[(lenZ+gap):yz,(lenX+gap):xz] = copy.copy(np.transpose(maxX[i,0]))

        imCh2[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i,1],axis=0))
        imCh2[(lenZ+gap):yz,0:lenX] = copy.copy(maxZ[i,1])
        imCh2[(lenZ+gap):yz,(lenX+gap):xz] = copy.copy(np.transpose(maxX[i,1]))
        
        contrastedImCh1 = adjustContrast(imCh1, adjMax, adjMin)
        contrastedImCh2 = adjustContrast(imCh2, adjMax, adjMin)
        frame = cv2.merge((contrastedImCh1,contrastedImCh2,contrastedImCh1))

        frame[np.where(imCh1==0)] = [0,0,0]
        
        # time stamp
        t = f'{i*IMAGING_FREQ // 60:02d}' + ':' + f'{i*IMAGING_FREQ % 60:02d}'
        #cv2.putText(frame,t,(25,lenZ+gap+150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)
        cv2.putText(frame,t,(15,lenZ+gap+30),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,255],3,cv2.LINE_AA)

        # add scale bars
        scaleBarXY = scaleBar(
            posY = yz - 20, #76
            posX = lenX - 50, #468
            height = 10, #30
            length = 42, #416
            text = '100 um', #1 mm
            textOffset = 0, #50
        )
        scaleBarXY._addScaleBar(frame)

        scaleBarXZ = scaleBar(
            posY = lenZ,
            posX = lenX + gap,
            height = 10, #30
            length = lenZ,
            text = str(lenZ * 2) + ' um',
            textOffset = 5, #50
        )
        scaleBarXZ._addScaleBarZ(frame)

        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

