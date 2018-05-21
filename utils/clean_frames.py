# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import h5py

h5name = '../datasets/annot.h5'
f = h5py.File(h5name, 'r')

id_list = ["TS1","TS2","TS3","TS4","TS5","TS6"]
for id_ in id_list:
  valid_list = list(f[id_]['valid_frame'])
  for idx in range(len(valid_list)):
    if valid_list[idx] == 0:
      path = '../datasets/test/'+id_+'/img_{}.jpg'.format(str(idx).zfill(6))
      if os.path.exists(path):
        os.remove(path)

f.close()
print("FINISH.")