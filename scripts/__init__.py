# Scripts package initialization
# This makes the directory importable as a Python package

# Define what gets imported when doing 'from scripts import *'
__all__ = ['OpticalFlow']

# Import frequently used modules for convenience
# You can add more here as needed
try:
    from . import OpticalFlow
except ImportError:
    pass  # This allows the script to be run standalone without causing errors
