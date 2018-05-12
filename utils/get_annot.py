# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import scipy.io
import h5py

DEBUG = False
DATA_PATH = '../../mpi_inf_3dhp/datasets/'
SAVE_PATH = '../datasets/'
if not os.path.exists(SAVE_PATH):
	os.mkdir(SAVE_PATH)

h5name = SAVE_PATH + 'annot.h5'
f = h5py.File(h5name, 'w')

id_list_ = ["S1","S2","S3","S4","S5","S6","S7","S8"]
for id_ in id_list_:
	mat = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/'+id_+'/Seq1/annot.mat')
	#mat['annot3'][camera][0][frame][jointx3]
	frames_1_ = len(mat['frames'])
	annot3_1_ = mat['annot3']
	annot2_1_ = mat['annot2']

	mat2 = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/'+id_+'/Seq2/annot.mat')
	frames_2_ = len(mat2['frames'])
	annot3_2_ = mat2['annot3']
	annot2_2_ = mat2['annot2']

	grp=f.create_group(id_)

	adict=dict(frames_1=frames_1_, frames_2=frames_2_)
	for i in range(len(annot3_1_)):
		adict['annot3_1_'+str(i)] = annot3_1_[i,0]
		adict['annot3_2_'+str(i)] = annot3_2_[i,0]
		adict['annot2_1_'+str(i)] = annot2_1_[i,0]
		adict['annot2_2_'+str(i)] = annot2_2_[i,0]
	for k,v in adict.items():
		grp.create_dataset(k,data=v)
f.close()

if DEBUG:
  g = h5py.File(h5name, 'r')
  print(g["S1"]["annot3_1_0"])

print("FINISH.")