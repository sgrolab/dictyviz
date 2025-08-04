# DictyViz package initialization
# This makes the directory importable as a Python package

# Import the most commonly used components for easier access
from .dictyviz import (
    channel,
    getChannelsFromJSON,
    getVoxelDimsFromXML,
    makeOrthoMaxMovie,
    makeOrthoMaxOpticalFlowVideo,
    makeSlicedMaxMovie,
    makeZStackMovie
)
