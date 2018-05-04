#!/bin/bash

for j in imageSequence/*.avi;
  do cam=`echo $j | cut -d'.' -f1 | cut -d'_' -f2`;
  echo $cam;
  ffmpeg -i "$j" -vf scale=320:320 "test/S1_1_${cam}_%05d.jpg"
done
