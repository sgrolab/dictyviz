# Dicty data functions for ome-zarr datasets

import os, zarr, cv2, cmapy, copy, math
import numpy as np
from tqdm import tqdm

def createRootStore(zarr_file):
    nestedStore = zarr.NestedDirectoryStore(zarr_file, dimension_separator='/')
    root = zarr.group(store=nestedStore, overwrite=False)

def createAnalysisGroup(root):

    if 'analysis' in root:
        anlaysisGroup = root['analysis']
    else:
        analysisGroup = root.create_group('analysis')

def getDimensions(resArray):
    return resArray.shape[0], resArray.shape[2], resArray.shape[3], resArray.shape[4]

def calcMaxProjections(root, nChannel, res_lvl=0):

    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenZ, lenY, lenX = getDimensions(resArray)
    
    analysisGroup = root['analysis']

    # create max projections group
    maxProjectionsGroup = analysisGroup.create_group('max_projections')

    # create zarr arrays for each max projection 
    maxZ = maxProjectionsGroup.zeros('maxz',shape=(lenT,lenY,lenX),chunks=(1,64,64))
    maxX = maxProjectionsGroup.zeros('maxx',shape=(lenT,lenZ,lenY),chunks=(1,64,64))
    maxY = maxProjectionsGroup.zeros('maxy',shape=(lenT,lenZ,lenX),chunks=(1,64,64))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(lenT)):
        frame = resArray[i, nChannel, :, :, :]
        maxZ[i] = np.max(frame,axis=0)
        maxX[i] = np.max(frame,axis=2)
        maxY[i] = np.max(frame,axis=1)

def calcSlicedMaxProjections(root, nChannel, res_lvl=0):
    # define resolution level
    resArray = root['0'][str(res_lvl)]

    # get dataset dimensions
    lenT, lenZ, lenY, lenX = getDimensions(resArray)

    analysisGroup = root['analysis']

    # create max projections group
    slicedMaxProjectionsGroup = analysisGroup.create_group('sliced_max_projections')

    # set number of slices
    sliceDepth = 83 # 83px*2.41um/px = 200 um
    nSlicesX = math.ceil(lenX/sliceDepth)
    nSlicesY = math.ceil(lenY/sliceDepth)

    # create zarr arrays for each max projection
    slicedMaxX = slicedMaxProjectionsGroup.zeros('sliced_maxx',shape=(nSlicesX,lenT,lenZ,lenY),chunks=(1,1,64,64))
    slicedMaxY = slicedMaxProjectionsGroup.zeros('sliced_maxy',shape=(nSlicesY,lenT,lenZ,lenX),chunks=(1,1,64,64))

    for i in range(nSlicesX):
        for j in range(lenT):
            rangeX = [i*sliceDepth, (i+1)*sliceDepth]
            frame = resArray[j, nChannel, :, :, rangeX[0]:rangeX[1]]
            slicedMaxX[i,j] = np.max(frame,axis=2)
    
    for i in range(nSlicesY):
        for j in range(lenT):
            rangeY = [i*sliceDepth, (i+1)*sliceDepth]
            frame = resArray[j, nChannel, :, rangeY[0]:rangeY[1], :]
            slicedMaxY[i,j] = np.max(frame,axis=1)

def createMoviesGroup(root):
    if 'movies' in root:
        moviesGroup = root['movies']
    else:
        moviesGroup = root.create_group('movies')

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
    def __init__(self, posY, posX, height, length, text, testOffset):
        self.posY = posY
        self.posX = posX
        self.height = height
        self.length = length
        self.text = text
        self.textOffset = textOffset

    def _addScaleBar(self, frame):
        frame[self.posY:self.posY+self.height, self.posX:self.posX+self.length,:] = 255
        cv2.putText(frame, self.text, (self.posX+self.textOffset, self.posY-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, [255,255,255], 6, cv2.LINE_AA)

    def _addScaleBarZ(self, frame):
        frame[self.posY:self.posY+self.height, self.posX:self.posX+self.length,:] = 255
        cv2.putText(frame, self.text, (self.posX, self.posY-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 3, cv2.LINE_AA)

def makeOrthoMaxVideo(filename, root, scaleMax):
    # define resolution level
    resArray = root['0'][str(res_lvl)]

    maxZ = root['analysis']['max_projections']['maxz']
    maxY = root['analysis']['max_projections']['maxy']
    maxX = root['analysis']['max_projections']['maxx']
    
    
    lenT = maxZ.shape[0]
    lenZ = maxY.shape[1]
    lenY = maxZ.shape[1]
    lenX = maxZ.shape[2]
    
    gap = 20

    xz = lenX + lenZ + gap
    yz = lenY + lenZ + gap

    imagingFreq = 10

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(xz,yz),1)
    #adjMax = 10000
    #adjust contrast based on max pixel value in the array
    adjMax = scaleMax
    adjMin = 0

    for i in tqdm(range(lenT)):

        # initialize frame 
        im = np.zeros([yz,xz])

        # copy max projections 
        im[0:lenZ,0:lenX] = copy.copy(np.flip(maxY[i],axis=0))
        im[(lenZ+gap):yz,0:lenX] = copy.copy(maxZ[i])
        im[(lenZ+gap):yz,(lenX+gap):xz] = copy.copy(np.transpose(maxx[i]))
        
        # adjust contrast
        # im[np.where(im>adjMax)] = adjMax
        # im[np.where(im<adjMin)] = adjMin
        # backSub = im - adjMin
        # backSub[np.where(backSub<0)] = 0
        # scaledIm = np.divide(backSub,adjMax-adjMin)
        # im8 = np.multiply(scaledIm,255).astype('uint8')
        contrastedIm = adjustContrast(im, adjMax, adjMin)
        frame = cv2.applyColorMap(contrastedIm,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]
        
        # time stamp
        t = f'{i*imagingFreq // 60:02d}' + ':' + f'{i*imagingFreq % 60:02d}'
        cv2.putText(frame,t,(25,lenZ+gap+150),cv2.FONT_HERSHEY_SIMPLEX,6,[255,255,255],10,cv2.LINE_AA)

        # add scale bars
        # TODO: use function that defines scale bar parameters using scale bar class
        scaleBarXY = scaleBar(
            posY = yz - 76,
            posX = lenX - 468,
            height = 30,
            length = 416,
            text = '1 mm',
            textOffset = 50,
        )
        scaleBarXY._addScaleBar(frame)

        scaleBarXZ = scaleBar(
            posY = lenZ,
            posX = lenX + gap,
            height = 30,
            length = lenZ,
            text = str(lenZ * 2) + ' um',
            textOffset = 50,
        )
        scaleBarXZ._addScaleBarZ(frame)

        # frame[scaleBarY:scaleBarY+scaleBarHeight,scaleBarX:scaleBarX+scaleBarLength,:] = 255
        # cv2.putText(frame,scaleBarText,(scaleBarX+scaleBarText_offset,scaleBarY-30),cv2.FONT_HERSHEY_SIMPLEX,3,[255,255,255],6,cv2.LINE_AA)
        
        # add xz scale bar (use scaleBar.addScaleBarZ)
        # frame[lenZ-scaleBarHeight:lenZ,lenX+gap:lenX+gap+lenZ,:] = 255
        # cv2.putText(frame,scaleBarZText,(lenX+gap,lenZ-50),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,255],3,cv2.LINE_AA)
        
        # write frame 
        vid.write(frame)

    vid.release()
    cv2.destroyAllWindows()

def makeSlicedOrthoMaxVideos(filename, z, scaleMax):
        
    slicedMaxX = z['analysis']['sliced_max_projections']['sliced_maxx']

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
            im[(lenZ*j+gap*j):(lenZ*(j+1)+gap*j),:] = copy.copy(np.flip(slicedMaxX[j,i], axis=0))

        # adjust contrast
        contrastedIm = adjustContrast(im, adjMax, adjMin)
        frame = cv2.applyColorMap(contrastedIm,cmapy.cmap('viridis'))

        frame[np.where(im==0)] = [0,0,0]

        # add time stamp
        t = f'{i*imagingFreq // 60:02d}' + ':' + f'{i*imagingFreq % 60:02d}'
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

            



