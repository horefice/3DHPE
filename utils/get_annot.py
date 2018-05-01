import numpy as np
import os
import scipy.io
import h5py

DEBUG = False
DATA_PATH = '../../mpi_inf_3dhp/datasets/'
SAVE_PATH = '../../mpi_inf_3dhp/datasets/'
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

mat = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/S1/Seq1/annot.mat')
id_ = "S1"
frames_ = len(mat['frames'])
annot3_ = mat['annot3']
annot2_ = mat['annot2']
#mat['annot3'][camera][0][frame][jointx3]

mat2 = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/S1/Seq2/annot.mat')
frames_ += +len(mat2['frames'])

h5name = SAVE_PATH + 'annot.h5'
f = h5py.File(h5name, 'w+')
grp=f.create_group('S1')

adict=dict(frames=frames_)
for i in range(len(annot3_)):
  adict['annot3_'+str(i)] = np.vstack((annot3_[i,0],mat2['annot3'][i,0]))
  adict['annot2_'+str(i)] = np.vstack((annot2_[i,0],mat2['annot2'][i,0]))
for k,v in adict.items():
  grp.create_dataset(k,data=v)
f.close()

if DEBUG:
  g = h5py.File(h5name, 'r')
  print(g["S1"]["annot3_0"])

print("FINISH.")