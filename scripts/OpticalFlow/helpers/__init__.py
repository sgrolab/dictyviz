# OpticalFlow helpers package initialization
# This makes the directory importable as a Python package

# Import the modules for easier access
from . import analyzeRegions
from . import flowLoader

# Expose commonly used functions
from .analyzeRegions import calculate_mag_var, find_optimal_regions, save_analysis_results
from .flowLoader import load_flow, normalize_flow
