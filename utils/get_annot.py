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

# TRAIN
id_list = ["S1","S2","S3","S4","S5","S6","S7","S8"]
for id_ in id_list:
  mat = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/'+id_+'/Seq1/annot.mat')
  #mat['univ_annot3'][camera][0][frame][jointx3]
  frames_1 = len(mat['frames'])
  annot3_1 = mat['univ_annot3']
  annot2_1 = mat['annot2']

  mat2 = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/'+id_+'/Seq2/annot.mat')
  frames_2 = len(mat2['frames'])
  annot3_2 = mat2['univ_annot3']
  annot2_2 = mat2['annot2']

  grp = f.create_group(id_)
  adict = dict(frames_1=frames_1, frames_2=frames_2)
  for i in range(len(annot3_1)):
    adict['annot3_1_'+str(i)] = annot3_1[i,0]
    adict['annot3_2_'+str(i)] = annot3_2[i,0]
    adict['annot2_1_'+str(i)] = annot2_1[i,0]
    adict['annot2_2_'+str(i)] = annot2_2[i,0]
  for k,v in adict.items():
    grp.create_dataset(k,data=v)

# TEST
id_list = ["TS1","TS2","TS3","TS4","TS5","TS6"]
for id_ in id_list:
  mat = h5py.File('../../mpi_inf_3dhp/datasets/mpi_inf_3dhp_test_set/'+id_+'/annot_data.mat', 'r')
  activity_annotation = mat['activity_annotation']
  univ_annot3 = mat['univ_annot3']
  valid_frame = mat['valid_frame']

  grp = f.create_group(id_)
  adict = dict(activity_annotation=activity_annotation, annot3=univ_annot3, valid_frame=valid_frame)
  for k,v in adict.items():
    grp.create_dataset(k,data=v)

f.close()
print("FINISH.")
