"""
dictyviz - Tools for generating orthogonal maximum intensity projections and movies 
from 4D zarr imaging datasets.
"""

__version__ = "0.1.0"

# Expose submodules for direct access
from . import visualization
from . import utils

# Expose main classes at top level for convenience
from .visualization import Channel, ScaleBar

__all__ = ["visualization", "utils", "Channel", "ScaleBar", "__version__"]
