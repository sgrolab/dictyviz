# OpticalFlow 2D package initialization
# This makes the directory importable as a Python package

# Import the modules for easier access
from . import opticalFlow
from . import slicedOpticalFlow
from . import testParameters

# Expose commonly used functions
from .opticalFlow import compute_farneback_optical_flow, make_movie
from .slicedOpticalFlow import compute_sliced_optical_flow
