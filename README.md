# Dictyviz

This repository provides a set of tools to generate orthogonal maximum intensity projections, movies from 4D zarr imaging datasets, and perform 3D optical flow analysis using a dedicated submodule.

---

## Getting Started

### 1. Clone the dictyviz repo

After cloning the repository, download the contents of the submodule:

```bash
git submodule update --init --recursive
```

### 2. Install the dictyviz environment 

```bash
conda env create -f environment.yml
conda activate dictyviz
```

### 3. Install the optical3dflow environment
The 3D optical flow analysis relies on a separate environment with additional dependencies located inside the submodule. To install it, first ensure you have the submodule content downloaded (see step 1).

```bash
conda env create -f optical3dflow_environment.yml
conda activate optical3dflow
```
### 4. Running scripts 
To run scripts, navigate to the right directory and run the correct bash script. Note: to run any optical flow calculations and findRegions.py, the optical3dflow environment must be activated. For everything else, the dictyviz environment should be activated. 
