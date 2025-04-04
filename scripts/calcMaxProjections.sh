#/bin/bash

bsub -n 8 -W 24:00 python calcOrthoMaxProjs.py
bsub -n 8 -W 24:00 python calcSlicedOrthoMaxProjs.py