# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import h5py

h5name = '../datasets/annot.h5'
f = h5py.File(h5name, 'r')
print(f["S1"]["annot3_1_0"][670][4*3:6*3])
print(f["TS5"]["annot3"][0,0,:,:])
print(list(f["TS5"]['valid_frame']).count(1))
f.close()