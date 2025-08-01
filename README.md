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
To run scripts, navigate to the right directory and run the correct bash script. Note: the two files that need the optical3dflow environment to be activated is when running the 3Dflow.py file with the generate3Dflow.sh bash script and the findRegions.py file with the generateFindRegions.sh bash script. 
