#!/bin/bash

# TRAIN
for k in ../../mpi_inf_3dhp/datasets/S*/; do
  subj=`echo $k | cut -d'S' -f2 | cut -d'/' -f1`;
  #echo S$subj;
  for j in {1..2}; do
    path=../datasets/train/S${subj}_${j}
    #echo $path;
    mkdir -p ${path}
    for i in ${k}Seq${j}/imageSequence/*.avi; do
      #echo $i;
      cam=`echo $i | cut -d'_' -f4 | cut -d'.' -f1`;
      ffmpeg -i ${i} -vf scale=320:320 ${path}/S${subj}_${j}_${cam}_%05d.jpg
    done
  done
done

# TEST
for k in ../../mpi_inf_3dhp/datasets/mpi_inf_3dhp_test_set/TS*/; do
  subj=`echo $k | cut -d'S' -f2 | cut -d'/' -f1`;
  #echo TS$subj;
  path=../datasets/test/TS${subj}
  #echo $path;
  mkdir -p ${path}
  for j in ${k}imageSequence/**; do
    cp ${j} ${path}
  done
done
